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

# Function to predict the location based on the maximum intensity in the heatmap
def predict_location_fast(heatmap):
    """Fastest location estimation using the max intensity point."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0  # If the max value in the heatmap is 0, return (0, 0)
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y, 1, 1  # Width and height are 1 (dummy value for calculating center)

# Function to predict coordinates (using inpainted coordinates or y_pred)
def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """Optimized version of coordinate prediction when using inpainted coordinates (c_pred)."""
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()  # Convert indices to numpy if tensor

    if c_pred is not None and isinstance(c_pred, torch.Tensor):
        c_pred = c_pred.detach().cpu().numpy()  # Convert c_pred to numpy if tensor

    # Ensure indices are in the correct shape (N, L, 2)
    if indices.ndim == 2:
        indices = np.expand_dims(indices, axis=1)  # Add extra dimension if necessary

    if c_pred is not None:
        if c_pred.ndim == 1:
            c_pred = c_pred.reshape(1, 1, 2)  # Reshape c_pred if necessary
        elif c_pred.ndim == 2:
            c_pred = c_pred[:, None, :]  # Add "L" dimension in c_pred

        # Handling inpainted coordinates
        N, L, _ = c_pred.shape
        flat_coords = c_pred.reshape(-1, 2)  # Flatten coordinates
        frames = indices[:, :, 1].reshape(-1)  # Flatten frame indices

        # Remove duplicates in frames (keep only first occurrence)
        _, unique_idx = np.unique(frames, return_index=True)
        flat_coords = flat_coords[unique_idx]
        frames = frames[unique_idx]

        # Resize coordinates to match image size
        cx = (flat_coords[:, 0] * WIDTH * img_scaler[0]).astype(int)
        cy = (flat_coords[:, 1] * HEIGHT * img_scaler[1]).astype(int)

        # Calculate visibility (0 if (0, 0), else 1)
        vis = ~((cx == 0) & (cy == 0)).astype(int)

        return {
            'Frame': frames.tolist(),
            'X': cx.tolist(),
            'Y': cy.tolist(),
            'Visibility': vis.tolist()
        }

    elif y_pred is not None:
        # If c_pred is not available, use y_pred
        pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        y_pred = y_pred > 0.5  # Apply threshold to decide if object is present
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # Reformat predictions to the expected format

        prev_f_i = -1
        for n in range(indices.shape[0]):
            for f in range(indices.shape[1]):
                f_i = indices[n][f][1]
                if f_i != prev_f_i:
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))  # Fast location prediction via heatmap

                    # Calculate x, y coordinates of the bounding box center
                    cx_pred = int((bbox_pred[0] + bbox_pred[2] / 2) * img_scaler[0])
                    cy_pred = int((bbox_pred[1] + bbox_pred[3] / 2) * img_scaler[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1  # Visibility calculation (non-zero means visible)

                    pred_dict['Frame'].append(int(f_i))
                    pred_dict['X'].append(cx_pred)
                    pred_dict['Y'].append(cy_pred)
                    pred_dict['Visibility'].append(vis_pred)
                    prev_f_i = f_i
                else:
                    break
        return pred_dict

    else:
        raise ValueError("Either y_pred or c_pred must be provided")  # Raise error if neither is provided

def main():
    # Argument parsing for command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='File path of the video')
    parser.add_argument('--tracknet_file', type=str, help='File path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='File path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='Evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='Maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='Range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='Directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='Whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='Whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='Length of trajectory to draw on video')
    args = parser.parse_args()

    # Initialize parameters
    num_workers = args.num_workers
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load TrackNet model
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    # Load InpaintNet if available
    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None  # If no InpaintNet model, don't use it

    # Open the video
    cap = cv2.VideoCapture(args.video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)
    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [], 'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    tracknet.eval()  # Set TrackNet model to evaluation mode
    seq_len = tracknet_seq_len

    # Allocate memory for buffers
    y_pred_buffer = torch.zeros((seq_len - 1, seq_len, HEIGHT, WIDTH), dtype=torch.float32, device='cuda')
    weight = get_ensemble_weight(seq_len, args.eval_mode).cuda()

    # Process video (large or small video)
    if large_video:
        dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                        max_sample_num=args.max_sample_num, video_range=video_range)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 drop_last=False, num_workers=num_workers,
                                 pin_memory=True, prefetch_factor=4, persistent_workers=True)
        video_len = dataset.video_len
        print(f'Video length: {video_len}')
    else:
        frame_list = generate_frames(args.video_file)  # Generate frames for small videos
        frame_arr = np.array(frame_list)
        if frame_arr.ndim < 4:
            frame_arr = np.stack(frame_list, axis=0)
        frame_arr = frame_arr[:, :, :, ::-1]  # Flip channels
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1,
                                                 data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=frame_arr)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False,
                                 pin_memory=True, prefetch_factor=4, persistent_workers=True)
        video_len = len(frame_list)

    # Run inference on video
    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().cuda()
        with torch.inference_mode():
            y_pred = tracknet(x).detach()  # TrackNet inference

        # Preallocate buffers and avoid resizing inside loops
        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)  # Append predictions to buffer
        # More operations...
        
    # Final saving operations
    # Write the results to CSV, output video, etc.
    print('Done.')

if __name__ == '__main__':
    start_time = time.time()
    import cProfile, pstats
    cProfile.run('main()', 'profiling_result')
    p = pstats.Stats('profiling_result')
    p.sort_stats('cumtime').print_stats(30)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
