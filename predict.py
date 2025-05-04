import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader, get_worker_info
from dataset import CircularVideoDataset
from utils.general import get_model, HEIGHT, WIDTH,write_pred_csv, write_pred_video,predict_location, get_ensemble_weight,predict
import time
from tqdm import tqdm



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    args = parser.parse_args()


    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file)
    seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])
    print(tracknet_ckpt['param_dict'])


    cap = cv2.VideoCapture(args.video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()

      
    dataset = CircularVideoDataset(
            video_file=video_file,
            seq_len=seq_len,
            sliding_step=1,
            bg_mode=bg_mode,
            HEIGHT=HEIGHT,
            WIDTH=WIDTH,
        )
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,              # 1 worker suffit souvent pour un IterableDataset
            pin_memory=True,
            prefetch_factor=2
        )

    video_len = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video length: {video_len}')


    

    # Initialisation des paramètres du buffer de prédiction
    num_sample = video_len - seq_len + 1
    sample_count = 0
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)              # [0, 1, 2, ..., seq_len-1]
    frame_i = torch.arange(seq_len - 1, -1, -1)   # [seq_len-1, ..., 0]
    y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
    weight = get_ensemble_weight(seq_len, 'weight')

    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.float().cuda()
        b_size, seq_len = i.shape[0], i.shape[1]
        
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()
        
        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
        ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
        ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

        for b in range(b_size):
            if sample_count < buffer_size:
                # Buffer incomplet
                y_pred_ens = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
            else:
                # Cas général avec pondération
                y_pred_ens = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

            ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
            sample_count += 1

            if sample_count == num_sample:
                # Dernier batch à traiter
                y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                for f in range(1, seq_len):
                    y_pred_ens = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                    ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

        # Prédiction
        tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

        # Mise à jour du buffer : conserver uniquement les dernières prédictions
        y_pred_buffer = y_pred_buffer[-buffer_size:]
    

    # Write csv file
    write_pred_csv(tracknet_pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
         write_pred_video(video_file, tracknet_pred_dict, save_file=out_video_file, traj_len=args.traj_len)

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
