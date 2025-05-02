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
    tracknet = tracknet.half().cuda().eval()

    
    seq_len = tracknet_seq_len
   
    # Préparation du dataset et du DataLoader en fonction de la taille de la vidéo
    if large_video:
        # Utilise un dataset itérable pour gérer de grandes vidéos par séquences
        dataset = Video_IterableDataset(
            video_file,
            seq_len=seq_len,
            sliding_step=1,
            bg_mode=bg_mode,
            max_sample_num=args.max_sample_num,
            video_range=video_range
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        video_len = dataset.video_len
        print(f'Video length: {video_len}')
    else:
        # Charger toutes les images en mémoire pour les petites vidéos
        frame_list = generate_frames(args.video_file)
        frame_arr = np.array(frame_list)
        # S'assurer que les dimensions sont correctes
        if frame_arr.ndim < 4:
            frame_arr = np.stack(frame_list, axis=0)
        # Convertir BGR en RGB
        frame_arr = frame_arr[:, :, :, ::-1]
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode='heatmap',
            bg_mode=bg_mode,
            frame_arr=frame_arr
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        video_len = len(frame_list)

    # Initialisation du buffer des prédictions intermédiaires sur GPU
    y_pred_buffer = torch.zeros(
        (seq_len - 1, seq_len, HEIGHT, WIDTH),
        dtype=torch.float32,
        device='cuda'
    )

    # Poids pour l'assemblage des prédictions selon le mode d'évaluation
    weight = get_ensemble_weight(seq_len, args.eval_mode).cuda()

    sample_count = 0
    # Indices pré-calculés pour gérer la fenêtre glissante dans le buffer
    batch_i = torch.arange(seq_len, device='cuda')
    frame_i = torch.arange(seq_len - 1, -1, -1, device='cuda')

    # Listes temporaires pour stocker indices et prédictions de l'ensemble
    ensemble_i_list = []
    ensemble_y_pred_list = []

    # Boucle d'inférence principale
    for step, (i, x) in enumerate(tqdm(data_loader)):
        x = x.half().cuda()
        # Désactive le calcul des gradients pour accélérer l'inférence
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda'):
                y_pred = tracknet(x) 
           

        b_size, seq_len_actual = i.shape[0], i.shape[1]
        # Ajoute les nouvelles prédictions au buffer
        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)

        for b in range(b_size):
            # Moyenne simple tant que le buffer n'est pas plein
            if sample_count < (seq_len - 1):
                y_pred_cur = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
            else:
                # Moyenne pondérée avec les poids d'ensemble
                y_pred_cur = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

            # Sauvegarde de l'indice de frame et de la prédiction correspondante
            ensemble_i_list.append(i[b][0].unsqueeze(0))
            ensemble_y_pred_list.append(y_pred_cur.unsqueeze(0))
            sample_count += 1

            # Gestion de la fin de la vidéo : padding et génération des prédictions manquantes
            if sample_count == (video_len - seq_len + 1):
                # Padding avec des zéros pour conserver la taille du buffer
                y_zero_pad = torch.zeros(
                    (seq_len - 1, seq_len_actual, HEIGHT, WIDTH),
                    dtype=torch.float32,
                    device='cuda'
                )
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                # Calcul des prédictions pour les frames de padding
                for f in range(1, seq_len_actual):
                    y_pred_cur = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len_actual - f)
                    ensemble_i_list.append(i[-1][f].unsqueeze(0))
                    ensemble_y_pred_list.append(y_pred_cur.unsqueeze(0))

        # Concaténation des résultats pour ce batch
        ensemble_i = torch.cat(ensemble_i_list, dim=0).cpu()
        ensemble_y_pred = torch.cat(ensemble_y_pred_list, dim=0).cpu()

        # Si les prédictions sont en 3D, ajoute un canal pour obtenir un tenseur 4D
        if ensemble_y_pred.dim() == 3:
            ensemble_y_pred = ensemble_y_pred.unsqueeze(1)

        # Génère les résultats finaux à partir des tenseurs préparés
        tmp_pred = predict(
            ensemble_i,
            y_pred=ensemble_y_pred,
            img_scaler=img_scaler
        )

        # Agrège les prédictions dans le dictionnaire global
        for key in tmp_pred.keys():
            tracknet_pred_dict[key].extend(tmp_pred[key])

        # Réinitialisation des listes et trimming du buffer pour le batch suivant
        ensemble_i_list = []
        ensemble_y_pred_list = []
        y_pred_buffer = y_pred_buffer[-(seq_len - 1):]




 # --- Inference avec Inpaint ---

    if inpaintnet is not None:
        inpaintnet = inpaintnet.half().cuda().eval()

        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        # Création du dataset pour l’inférence du réseau d’inpainting en mode 'coordinate'
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode='coordinate',
            pred_dict=tracknet_pred_dict
        )

        # Chargement des données avec DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        # Poids pour la pondération de l’ensemble selon le mode d’évaluation
        weight = get_ensemble_weight(seq_len, args.eval_mode).cuda()
        num_sample   = len(dataset)     # Nombre total d’échantillons
        sample_count = 0                # Compteur d’échantillons traités
        buffer_size  = seq_len - 1      # Taille de la fenêtre glissante

        # Indices pré-calculés pour extraire la fenêtre du buffer
        batch_i = torch.arange(seq_len, device='cuda')
        frame_i = torch.arange(seq_len - 1, -1, -1, device='cuda')

        # Buffer GPU pour stocker les coordonnées inpaintées au fil des batches
        coor_inpaint_buffer = torch.zeros(
            (buffer_size, seq_len, 2),
            dtype=torch.float32,
            device='cuda'
        )

        # Listes temporaires pour accumuler les résultats de l’ensemble
        ensemble_i_list             = []
        ensemble_coor_inpaint_list  = []

        # Boucle principale d’inférence en mode inpainting
        for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
            # Passage en FP16 + GPU des tenseurs d’entrée
            coor_pred    = coor_pred.half().cuda()
            inpaint_mask = inpaint_mask.half().cuda()

            # Génération des coordonnées inpaintées avec AMP
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    coor_inpaint = inpaintnet(coor_pred, inpaint_mask).detach()
                    # Reconstruire : conserver inpaint là où le masque est actif
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

            # Cloner pour permettre la modification sans altérer le buffer d’origine
            coor_inpaint = coor_inpaint.clone()

            # Filtrer les coordonnées de faible confiance
            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.

            # Ajouter les nouvelles prédictions au buffer (concaténation le long de la première dim.)
            coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)

            # Parcours de chaque échantillon du batch pour calculer l’ensemble
            b_size = i.shape[0]
            for b in range(b_size):
                if sample_count < buffer_size:
                    # Moyenne simple tant que le buffer n’est pas encore plein
                    coor_inpaint_cur = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    # Moyenne pondérée par les poids d’ensemble
                    coor_inpaint_cur = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                # Stocker l’indice et la coordonnée calculée
                ensemble_i_list.append(i[b][0].unsqueeze(0))
                ensemble_coor_inpaint_list.append(coor_inpaint_cur.unsqueeze(0))
                sample_count += 1

                # À la fin des échantillons : padding + calcul des dernières frames
                if sample_count == num_sample:
                    # Padding avec un buffer de zéros
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), device='cuda')
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                    # Calcul pour chaque frame manquant après padding
                    for f in range(1, seq_len):
                        coor_inpaint_cur = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i_list.append(i[-1][f].unsqueeze(0))
                        ensemble_coor_inpaint_list.append(coor_inpaint_cur.unsqueeze(0))

            # Concaténation et transfert en CPU pour la prédiction finale
            ensemble_i            = torch.cat(ensemble_i_list, dim=0).cpu()
            ensemble_coor_inpaint = torch.cat(ensemble_coor_inpaint_list, dim=0).cpu()

            # Si nécessaire, ajouter une dimension pour correspondre au format attendu
            if ensemble_coor_inpaint.dim() == 3:
                ensemble_coor_inpaint = ensemble_coor_inpaint.unsqueeze(1)

            # Conversion au format final via la fonction predict
            tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
            for key in tmp_pred:
                inpaint_pred_dict[key].extend(tmp_pred[key])

            # Réinitialiser pour le batch suivant
            ensemble_i_list            = []
            ensemble_coor_inpaint_list = []
            coor_inpaint_buffer        = coor_inpaint_buffer[-buffer_size:]



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