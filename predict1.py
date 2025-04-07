import os 
import argparse
import numpy as np
from tqdm import tqdm
import time
import sys


import torch
from torch.utils.data import DataLoader
import cv2

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *

"""def predict_location_hybrid(heatmap, use_contour_threshold=0.2):
    
    If the heatmap has a clear strong peak, use fast argmax.
    Otherwise fallback to cv2.findContours (original method).
    
    max_val = np.max(heatmap)
    if max_val == 0:
        return 0, 0, 0, 0
    if max_val >= use_contour_threshold * 255:  # scale is assumed to be 0–255
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return x, y, 1, 1
    else:
        return predict_location(heatmap)  """

def predict_location_fast(heatmap):
    """Fastest location estimation using the max intensity point."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y, 1, 1  # width and height = 1 (mock, but enough for center calc)



def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """
    Optimized version of coordinate prediction when using inpainted coordinates (c_pred).
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    if c_pred is not None and isinstance(c_pred, torch.Tensor):
        c_pred = c_pred.detach().cpu().numpy()

    # Ensure proper shape: (N, L, 2)
    if indices.ndim == 2:
        indices = np.expand_dims(indices, axis=1)
    if c_pred is not None:
        if c_pred.ndim == 1:
            c_pred = c_pred.reshape(1, 1, 2)
        elif c_pred.ndim == 2:
            c_pred = c_pred[:, None, :]

    if c_pred is not None:
        # Flatten everything
        N, L, _ = c_pred.shape
        flat_coords = c_pred.reshape(-1, 2)
        frames = indices[:, :, 1].reshape(-1)

        # Remove duplicates based on frame index (keep first occurrence)
        _, unique_idx = np.unique(frames, return_index=True)
        flat_coords = flat_coords[unique_idx]
        frames = frames[unique_idx]

        # Rescale coordinates
        cx = (flat_coords[:, 0] * WIDTH * img_scaler[0]).astype(int)
        cy = (flat_coords[:, 1] * HEIGHT * img_scaler[1]).astype(int)

        # Compute visibility
        vis = ~((cx == 0) & (cy == 0)).astype(int)

        return {
            'Frame': frames.tolist(),
            'X': cx.tolist(),
            'Y': cy.tolist(),
            'Visibility': vis.tolist()
        }

    elif y_pred is not None:
        # keep fallback logic for y_pred (non-vectorisable for now)
        # can be parallelized later with joblib or torch.jit
        pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)
        prev_f_i = -1
        for n in range(indices.shape[0]):
            for f in range(indices.shape[1]):
                f_i = indices[n][f][1]
                if f_i != prev_f_i:
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location_fast(to_img(y_p))  # ou predict_location_hybrid(...)

                    cx_pred = int((bbox_pred[0] + bbox_pred[2] / 2) * img_scaler[0])
                    cy_pred = int((bbox_pred[1] + bbox_pred[3] / 2) * img_scaler[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    pred_dict['Frame'].append(int(f_i))
                    pred_dict['X'].append(cx_pred)
                    pred_dict['Y'].append(cy_pred)
                    pred_dict['Visibility'].append(vis_pred)
                    prev_f_i = f_i
                else:
                    break
        return pred_dict

    else:
        raise ValueError("Either y_pred or c_pred must be provided")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for DataLoader')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    args = parser.parse_args()

    num_workers = args.num_workers
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])
    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    cap = cv2.VideoCapture(args.video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                          'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # --- Inference avec TrackNet ---
    tracknet.eval()
    seq_len = tracknet_seq_len
    if args.eval_mode == 'nonoverlap':
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False, num_workers=num_workers,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
            print(f'Video length: {dataset.video_len}')
        else:
            frame_list = generate_frames(args.video_file)
            frame_arr = np.array(frame_list)
            if frame_arr.ndim < 4:
                frame_arr = np.stack(frame_list, axis=0)
            frame_arr = frame_arr[:, :, :, ::-1]
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len,
                                                      data_mode='heatmap', bg_mode=bg_mode,
                                                      frame_arr=frame_arr, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.inference_mode():
                y_pred = tracknet(x).detach()  # Reste sur GPU
            # Transfert unique du batch sur CPU avant predict
            tmp_pred = predict(i, y_pred=y_pred.cpu(), img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     drop_last=False, num_workers=num_workers,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
        else:
            frame_list = generate_frames(args.video_file)
            frame_arr = np.array(frame_list)
            if frame_arr.ndim < 4:
                frame_arr = np.stack(frame_list, axis=0)
            frame_arr = frame_arr[:, :, :, ::-1]
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1,
                                                      data_mode='heatmap', bg_mode=bg_mode,
                                                      frame_arr=frame_arr)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
            video_len = len(frame_list)
        y_pred_buffer = torch.zeros((seq_len - 1, seq_len, HEIGHT, WIDTH),
                                    dtype=torch.float32, device='cuda')
        weight = get_ensemble_weight(seq_len, args.eval_mode).cuda()
        sample_count = 0
        batch_i = torch.arange(seq_len, device='cuda')
        frame_i = torch.arange(seq_len - 1, -1, -1, device='cuda')
        ensemble_i_list = []
        ensemble_y_pred_list = []
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.inference_mode():
                y_pred = tracknet(x).detach()  # Reste sur GPU
            b_size, seq_len_actual = i.shape[0], i.shape[1]
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            for b in range(b_size):
                if sample_count < (seq_len - 1):
                    y_pred_cur = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    y_pred_cur = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
                ensemble_i_list.append(i[b][0].unsqueeze(0))
                ensemble_y_pred_list.append(y_pred_cur.unsqueeze(0))
                sample_count += 1
                if sample_count == (video_len - seq_len + 1):
                    y_zero_pad = torch.zeros((seq_len - 1, seq_len_actual, HEIGHT, WIDTH),
                                             dtype=torch.float32, device='cuda')
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                    for f in range(1, seq_len_actual):
                        y_pred_cur = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len_actual - f)
                        ensemble_i_list.append(i[-1][f].unsqueeze(0))
                        ensemble_y_pred_list.append(y_pred_cur.unsqueeze(0))
            ensemble_i = torch.cat(ensemble_i_list, dim=0).cpu()
            ensemble_y_pred = torch.cat(ensemble_y_pred_list, dim=0).cpu()
            # Si ensemble_y_pred est 3D, ajouter une dimension pour obtenir un tenseur 4D
            if ensemble_y_pred.dim() == 3:
                ensemble_y_pred = ensemble_y_pred.unsqueeze(1)
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
            ensemble_i_list = []
            ensemble_y_pred_list = []
            y_pred_buffer = y_pred_buffer[-(seq_len - 1):]

    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        if args.eval_mode == 'nonoverlap':
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len,
                                                      data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False,
                                     pin_memory=True, prefetch_factor=4 , persistent_workers=True)
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float().cuda(), inpaint_mask.float().cuda()
                with torch.inference_mode():
                    coor_inpaint = inpaintnet(coor_pred, inpaint_mask).detach()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                tmp_pred = predict(i, c_pred=coor_inpaint.cpu(), img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
        else:
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1,
                                                      data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
            weight = get_ensemble_weight(seq_len, args.eval_mode).cuda()
            num_sample = len(dataset)
            sample_count = 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len, device='cuda')
            frame_i = torch.arange(seq_len - 1, -1, -1, device='cuda')
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2),
                                              dtype=torch.float32, device='cuda')
            ensemble_i_list = []
            ensemble_coor_inpaint_list = []
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float().cuda(), inpaint_mask.float().cuda()
                b_size = i.shape[0]
                with torch.inference_mode():
                    coor_inpaint = inpaintnet(coor_pred, inpaint_mask).detach()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                coor_inpaint = coor_inpaint.clone()  # Clone pour obtenir un tenseur modifiable
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                for b in range(b_size):
                    if sample_count < buffer_size:
                        coor_inpaint_cur = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        coor_inpaint_cur = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)
                    ensemble_i_list.append(i[b][0].unsqueeze(0))
                    ensemble_coor_inpaint_list.append(coor_inpaint_cur.unsqueeze(0))
                    sample_count += 1
                    if sample_count == num_sample:
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2),
                                                    dtype=torch.float32, device='cuda')
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        for f in range(1, seq_len):
                            coor_inpaint_cur = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i_list.append(i[-1][f].unsqueeze(0))
                            ensemble_coor_inpaint_list.append(coor_inpaint_cur.unsqueeze(0))
                ensemble_i = torch.cat(ensemble_i_list, dim=0).cpu()
                ensemble_coor_inpaint = torch.cat(ensemble_coor_inpaint_list, dim=0).cpu()
                if ensemble_coor_inpaint.dim() == 3:
                    ensemble_coor_inpaint = ensemble_coor_inpaint.unsqueeze(1)
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                ensemble_i_list = []
                ensemble_coor_inpaint_list = []
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]
    
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)
    if args.output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)
    print('Done.')

if __name__ == '__main__':
    

    start_time = time.time()
    import cProfile, pstats
    cProfile.run('main()', 'profiling_result')
    p = pstats.Stats('profiling_result')
    p.sort_stats('cumtime').print_stats(30)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Temps total d'exécution: {total_time:.2f} secondes")