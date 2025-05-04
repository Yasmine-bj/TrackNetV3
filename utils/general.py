
import json
import math
import parse
import shutil
import numpy as np
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from model import TrackNet
import cv2
from collections import deque
import imageio



# Global variables
HEIGHT = 288
WIDTH = 512
SIGMA = 2.5
DELTA_T = 1/math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH = DELTA_T * 50
IMG_FORMAT = 'png'



def get_ensemble_weight(seq_len, eval_mode):
    """Get weight for temporal ensemble.

    Args:
        seq_len (int): Length of input sequence
        eval_mode (str): Mode of temporal ensemble
            Choices:
                - 'average': Return uniform weight
                - 'weight': Return positional weight

    Returns:
        torch.Tensor: Weight for temporal ensemble
    """
    if eval_mode == 'average':
        return torch.full((seq_len,), 1.0 / seq_len, dtype=torch.float32)
    elif eval_mode == 'weight':
        # Créer un vecteur [0, 1, 2, ..., seq_len-1]
        indices = torch.arange(seq_len, dtype=torch.float32)
        # Calculer les poids symétriques en utilisant torch.min avec le vecteur inversé
        weight = torch.min(indices + 1, torch.flip(indices, dims=[0]) + 1)
        weight = weight / weight.sum()
        return weight
    else:
        raise ValueError('Invalid mode')


def predict_location(heatmap):
    """ Get coordinates from the heatmap.

        Args:
            heatmap (numpy.ndarray): A single heatmap with shape (H, W)

        Returns:
            x, y, w, h (Tuple[int, int, int, int]): bounding box of the the bounding box with max area
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0, 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        return x, y, w, h

def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.3
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    

###################################  Helper Functions ###################################
def get_model(model_name, seq_len=None, bg_mode=None):
    """ Create model by name and the configuration parameter.

        Args:
            model_name (str): type of model to create
                Choices:
                    - 'TrackNet': Return TrackNet model
                    - 'InpaintNet': Return InpaintNet model
            seq_len (int, optional): Length of input sequence of TrackNet
            bg_mode (str, optional): Background mode of TrackNet
                Choices:
                    - '': Return TrackNet with L x 3 input channels (RGB)
                    - 'subtract': Return TrackNet with L x 1 input channel (Difference frame)
                    - 'subtract_concat': Return TrackNet with L x 4 input channels (RGB + Difference frame)
                    - 'concat': Return TrackNet with (L+1) x 3 input channels (RGB)

        Returns:
            model (torch.nn.Module): Model with specified configuration
    """

    if model_name == 'TrackNet':
        if bg_mode == 'subtract':
            model = TrackNet(in_dim=seq_len, out_dim=seq_len)
        elif bg_mode == 'subtract_concat':
            model = TrackNet(in_dim=seq_len*4, out_dim=seq_len)
        elif bg_mode == 'concat':
            model = TrackNet(in_dim=(seq_len+1)*3, out_dim=seq_len)
        else:
            model = TrackNet(in_dim=seq_len*3, out_dim=seq_len)
    elif model_name == 'InpaintNet':
        model = InpaintNet()
    else:
        raise ValueError('Invalid model name.')
    
    return model


def to_img(image):
    """ Convert the normalized image back to image format.

        Args:
            image (numpy.ndarray): Images with range in [0, 1]

        Returns:
            image (numpy.ndarray): Images with range in [0, 255]
    """

    image = image * 255
    image = image.astype('uint8')
    return image





def to_img_format(input, num_ch=1):
    """ Helper function for transforming model input sequence format to image sequence format.

        Args:
            input (numpy.ndarray): model input with shape (N, L*C, H, W)
            num_ch (int): Number of channels of each frame.

        Returns:
            (numpy.ndarray): Image sequences with shape (N, L, H, W) or (N, L, H, W, 3)
    """

    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    
    if num_ch == 1:
        # (N, L, H ,W)
        return input
    else:
        # (N, L*C, H ,W)
        input = np.transpose(input, (0, 2, 3, 1)) # (N, H ,W, L*C)
        seq_len = int(input.shape[-1]/num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3) # (N, L, H, W, 3)
        # For each sample in the batch
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            # Get each frame in the sequence
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f+3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        
        return img_seq




def draw_traj(img, traj, radius=3, color='red'):
    """ Draw trajectory on the image.

        Args:
            img (numpy.ndarray): Image with shape (H, W, C)
            traj (deque): Trajectory to draw

        Returns:
            img (numpy.ndarray): Image with trajectory drawn
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    img = Image.fromarray(img)
    
    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox =  (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img



import threading
import queue
import cv2
from collections import deque

def write_pred_video(video_file, pred_dict, save_file, traj_len=8, queue_size=16):
    # 1) Ouvre la vidéo source
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2) Initialise le VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_file, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer VideoWriter pour {save_file}")

    # 3) Crée la queue et lance le thread d’écriture
    video_queue = queue.Queue(maxsize=queue_size)

    def write_frames():
        while True:
            frame = video_queue.get()
            if frame is None:
                break
            writer.write(frame)
            video_queue.task_done()
        writer.release()

    thread = threading.Thread(target=write_frames, daemon=True)
    thread.start()

    # 4) Parcours de la vidéo principale et push des frames annotées
    frames_pred = pred_dict['Frame']
    x_pred      = pred_dict['X']
    y_pred      = pred_dict['Y']
    vis_pred    = pred_dict['Visibility']
    pred_queue  = deque(maxlen=traj_len)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Met à jour la trajectoire
        if frame_index < len(vis_pred) and vis_pred[frame_index]:
            pred_queue.append((x_pred[frame_index], y_pred[frame_index]))
        else:
            pred_queue.append(None)

        # Dessine les points
        for pt in pred_queue:
            if pt is not None:
                cv2.circle(frame, pt, 3, (0,255,255), -1)

        # Push non bloquant dans la queue
        try:
            video_queue.put(frame, timeout=0.1)
        except queue.Full:
            # Si pleine, on drop pour ne pas bloquer
            pass

        frame_index += 1

    # 5) Terminaison propre
    cap.release()
    video_queue.put(None)    # signal de fin pour le thread
    thread.join()
    print(f"Vidéo annotée sauvegardée dans : {save_file}")



# def write_pred_video(video_file, pred_dict, save_file, traj_len=8, label_df=None):
#     """
#     Write a video with prediction result, using imageio-ffmpeg for H.264 output.
#     """
#     # Read input video to get fps & frame size
#     cap = cv2.VideoCapture(video_file)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Prepare output directory
#     os.makedirs(os.path.dirname(save_file), exist_ok=True)

#     # Open imageio writer with libx264 codec
#     writer = imageio.get_writer(
#         save_file,
#         fps = fps,
#         codec = 'libx264',
#         ffmpeg_params = ['-pix_fmt', 'yuv420p']  # assure la compatibilité
#     )

#     # Load prediction arrays
#     frames_pred = pred_dict['Frame']
#     x_pred      = pred_dict['X']
#     y_pred      = pred_dict['Y']
#     vis_pred    = pred_dict['Visibility']

#     # If ground-truth provided
#     if label_df is not None:
#         frames_gt = label_df['Frame'].tolist()
#         x_gt      = label_df['X'].tolist()
#         y_gt      = label_df['Y'].tolist()
#         vis_gt    = label_df['Visibility'].tolist()

#     # Queues for trajectories
#     pred_queue = deque(maxlen=traj_len)
#     if label_df is not None:
#         gt_queue = deque(maxlen=traj_len)

#     frame_index = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Append new point or None
#         if frame_index < len(vis_pred) and vis_pred[frame_index]:
#             pred_queue.append((x_pred[frame_index], y_pred[frame_index]))
#         else:
#             pred_queue.append(None)

#         if label_df is not None:
#             if frame_index < len(vis_gt) and vis_gt[frame_index]:
#                 gt_queue.append((x_gt[frame_index], y_gt[frame_index]))
#             else:
#                 gt_queue.append(None)

#         # Draw trajectories
#         # ground truth in red
#         if label_df is not None:
#             for pt in gt_queue:
#                 if pt is not None:
#                     cv2.circle(frame, pt, 3, (0,0,255), -1)

#         # predictions in yellow
#         for pt in pred_queue:
#             if pt is not None:
#                 cv2.circle(frame, pt, 3, (0,255,255), -1)

#         # Convert BGR→RGB for imageio
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         writer.append_data(rgb)

#         frame_index += 1

#     writer.close()
#     cap.release()
#     print(f"Vidéo annotée sauvegardée dans : {save_file}")

def write_pred_csv(pred_dict, save_file, save_inpaint_mask=False):
    """ Write prediction result to csv file.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame': frame_id (List[int]),
                         'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int]),
                         'Inpaint_Mask': inpaint_mask (List[int])}
            save_file (str): File path of the output csv file
            save_inpaint_mask (bool, optional): Whether to save inpaint mask

        Returns:
            None
    """

    if save_inpaint_mask:
        # Save temporary data for InpaintNet training
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility_GT': pred_dict['Visibility_GT'],
                                'X_GT': pred_dict['X_GT'],
                                'Y_GT': pred_dict['Y_GT'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'], 
                                'Y': pred_dict['Y'],
                                'Inpaint_Mask': pred_dict['Inpaint_Mask']})
    else:
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'],
                                'Y': pred_dict['Y']})
    pred_df.to_csv(save_file, index=False)
    

################################ Preprocessing Functions ################################
def generate_data_frames(video_file):
    """ Sample frames from the videos in the dataset.

        Args:
            video_file (str): File path of video in dataset
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
        
        Actions:
            Generate frames from the video and save as image files to the corresponding frame directory
    """

    # Check file format
    try:
        assert video_file[-4:] == '.mp4', 'Invalid video file format.'
    except:
        raise ValueError(f'{video_file} is not a video file.')

    # Check if the video has matched csv file
    file_format_str = os.path.join('{}', 'video', '{}.mp4')
    match_dir, rally_id = parse.parse(file_format_str, video_file)
    csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
    label_df = pd.read_csv(csv_file, encoding='utf8')
    assert os.path.exists(video_file) and os.path.exists(csv_file), 'Video file or csv file does not exist.'

    rally_dir = os.path.join(match_dir, 'frame', rally_id)
    if not os.path.exists(rally_dir):
        # Haven't processed yet
        os.makedirs(rally_dir)
    else:
        label_df = pd.read_csv(csv_file, encoding='utf8')
        if len(list_dirs(rally_dir)) < len(label_df):
            # Some error has occured, remove the directory and process again
            shutil.rmtree(rally_dir)
            os.makedirs(rally_dir)
        else:
            # Already processed.
            return

    cap = cv2.VideoCapture(video_file)
    frames = []
    success = True

    # Sample frames until video end or exceed the number of labels
    while success and len(frames) != len(label_df):
        success, frame = cap.read()
        if success:
            frames.append(frame)
            cv2.imwrite(os.path.join(rally_dir, f'{len(frames)-1}.{IMG_FORMAT}'), frame)
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)
    median = median[..., ::-1] # BGR to RGB
    np.savez(os.path.join(rally_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

