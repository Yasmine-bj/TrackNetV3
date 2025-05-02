import os
import cv2
import math
import parse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from utils.general import get_rally_dirs, get_match_median, HEIGHT, WIDTH, SIGMA, IMG_FORMAT
from line_profiler import LineProfiler 
import numpy as np
import time
from PIL import Image
import torch
data_dir = 'data'


class Shuttlecock_Trajectory_Dataset(Dataset):
    """ Shuttlecock_Trajectory_Dataset
            Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
    """
    def __init__(self,
        root_dir=data_dir,
        split='train',
        seq_len=8,
        sliding_step=1,
        data_mode='heatmap',
        bg_mode='',
        frame_alpha=-1,
        rally_dir=None,
        frame_arr=None,
        pred_dict=None,
        padding=False,
        debug=False,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        SIGMA=SIGMA,
        median=None
    ):
        """ Initialize the dataset

            Args:
                root_dir (str): File path of root directory of the dataset
                split (str): Split of the dataset, 'train', 'test' or 'val'
                seq_len (int): Length of the input sequence
                sliding_step (int): Sliding step of the sliding window during the generation of input sequences
                data_mode (str): Data mode
                    Choices:
                        - 'heatmap':Return TrackNet input data
                        - 'coordinate': Return InpaintNet input data
                bg_mode (str): Background mode
                    Choices:
                        - '': Return original frame sequence
                        - 'subtract': Return the difference frame sequence
                        - 'subtract_concat': Return the frame sequence with RGB and difference frame channels
                        - 'concat': Return the frame sequence with background as the first frame
                frame_alpha (float): Frame mixup alpha
                rally_dir (str): Rally directory
                frame_arr (numpy.ndarray): Frame sequence for TrackNet inference
                pred_dict (Dict): Prediction dictionary for InpaintNet inference
                    Format: {'X': x_pred (List[int]),
                             'Y': y_pred (List[int]),
                             'Visibility': vis_pred (List[int]),
                             'Inpaint_Mask': inpaint_mask (List[int]),
                             'Img_scaler': img_scaler (Tuple[int]),
                             'Img_shape': img_shape (Tuple[int])}
                padding (bool): Padding the last frame if the frame sequence is shorter than the input sequence
                debug (bool): Debug mode
                HEIGHT (int): Height of the image for input.
                WIDTH (int): Width of the image for input.
                SIGMA (int): Sigma of the Gaussian heatmap which control the label size.
                median (numpy.ndarray): Median image
        """

        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat'], f'Invalid bg_mode: {bg_mode}, should be "", subtract, subtract_concat or concat'

        # Image size
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        # Gaussian heatmap parameters
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        self.split = split if rally_dir is None else self._get_split(rally_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # Data for inference
        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len



        # Prétraitement par batch des images (déplacement ici dans la méthode)
        if self.frame_arr is not None:
            self.frame_arr_resized = np.array([np.array(Image.fromarray(frame).resize((WIDTH, HEIGHT))) for frame in self.frame_arr])
        
        # Initialize the input data
        if self.frame_arr is not None:
            # For TrackNet inference
            assert self.data_mode == 'heatmap', f'Invalid data_mode: {self.data_mode}, frame_arr only for heatmap mode' 
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()
            if self.bg_mode:
                if median is None:
                    median = np.median(self.frame_arr, 0)
                if self.bg_mode == 'concat':
                    median = Image.fromarray(median.astype('uint8'))
                    median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(median, -1, 0)
                else:
                    self.median = median
        elif self.pred_dict is not None:
            # For InpaintNet inference
            assert self.data_mode == 'coordinate', f'Invalid data_mode: {self.data_mode}, pred_dict only for coordinate mode'
            self.data_dict, self.img_config = self._gen_input_from_pred_dict()


    def _get_rally_i(self, rally_dir):
        """ Return the corresponding rally index of the rally directory. """
        if rally_dir not in self.rally_dict['p2i'].keys():
            return None
        else:
            return self.rally_dict['p2i'][rally_dir]

    def _get_split(self, rally_dir):
        """ Parse the split from the rally directory. """
        file_format_str = os.path.join(self.root_dir, '{}', 'match{}')
        split, _ = parse.parse(file_format_str, rally_dir)
        return split
    
    
            
 

    def _gen_input_from_frame_arr(self):
        """ Generate input sequences from a frame array. """

        # Calculate the image scaler
        h, w, _ = self.frame_arr[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        last_idx = -1
        for i in range(0, len(self.frame_arr), self.sliding_step):
            tmp_idx = []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(self.frame_arr):
                    tmp_idx.append((0, i+f))
                    last_idx = i+f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                    else:
                        break
            if len(tmp_idx) == self.seq_len:
                # Append the input sequence
                id = np.concatenate((id, [tmp_idx]), axis=0)
        
        return dict(id=id), dict(img_scaler=(w_scaler, h_scaler), img_shape=(w, h))

    def _gen_input_from_pred_dict(self):
        """ Generate input sequences from a prediction dictionary. """
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        x_pred, y_pred, vis_pred = self.pred_dict['X'], self.pred_dict['Y'], self.pred_dict['Visibility']
        inpaint = self.pred_dict['Inpaint_Mask']
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint), \
            f'Length of x_pred, y_pred, vis_pred and inpaint are not equal.'
        
        # Sliding on the frame sequence
        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i+f < len(inpaint):
                    tmp_idx.append((0, i+f))
                    tmp_coor_pred.append((x_pred[i+f], y_pred[i+f]))
                    tmp_vis_pred.append(vis_pred[i+f])
                    tmp_inpaint.append(inpaint[i+f])
                    last_idx = i+f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                    else:
                        break
                
            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint), \
                    f'Length of predicted coordinates and inpaint masks are not equal.'
                id = np.concatenate((id, [tmp_idx]), axis=0)
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)
        
        return dict(id=id, coor_pred=coor_pred, pred_vis=pred_vis, inpaint_mask=inpaint_mask),\
               dict(img_scaler=self.pred_dict['Img_scaler'], img_shape=self.pred_dict['Img_shape']) 
    
   
    def __len__(self):
        """ Return the number of data in the dataset. """
        return len(self.data_dict['id'])
    
    
    def __getitem__(self, idx):
        
        """Return the data of the given index.
        
        Pour training/évaluation:
        'heatmap': Return data_idx, frames, heatmaps, tmp_coor, tmp_vis
        'coordinate': Return data_idx, coor_pred, inpaint

        Pour inference:
        'heatmap': Return data_idx, frames
        'coordinate': Return data_idx, coor_pred, inpaint"""
        
        
        # --- Cas 1 : Les frames sont préchargées (inférence ou training sur frame_arr) ---
        if self.frame_arr is not None:
            data_idx = self.data_dict['id'][idx]  # (L,)
            imgs = self.frame_arr[data_idx[:, 1], ...]  # (L, H, W, 3)
            if self.bg_mode:
                median_img = self.median

            # Accumuler les frames dans une liste pour éviter des concaténations répétées
            frames_list = []
            for i in range(self.seq_len):
                img = Image.fromarray(imgs[i])
                if self.bg_mode == 'subtract':
                    proc_img = Image.fromarray(
                        np.sum(np.absolute(np.array(img) - median_img), axis=2).astype('uint8')
                    )
                    proc_img = np.array(proc_img.resize((self.WIDTH, self.HEIGHT))).reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    diff_img = Image.fromarray(
                        np.sum(np.absolute(np.array(img) - median_img), axis=2).astype('uint8')
                    )
                    diff_img = np.array(diff_img.resize((self.WIDTH, self.HEIGHT))).reshape(1, self.HEIGHT, self.WIDTH)
                    img_resized = np.array(img.resize((self.WIDTH, self.HEIGHT)))
                    img_resized = np.moveaxis(img_resized, -1, 0)
                    proc_img = np.concatenate((img_resized, diff_img), axis=0)
                else:
                    proc_img = np.array(img.resize((self.WIDTH, self.HEIGHT)))
                    proc_img = np.moveaxis(proc_img, -1, 0)
                frames_list.append(proc_img)
            # Concaténer toutes les frames une seule fois
            frames = np.concatenate(frames_list, axis=0)
            if self.bg_mode == 'concat':
                frames = np.concatenate((median_img, frames), axis=0)
            frames = frames / 255.
            return data_idx, frames

        # --- Cas 2 : Utilisation des prédictions déjà effectuées ---
        elif self.pred_dict is not None:
            data_idx = self.data_dict['id'][idx]  # (L,)
            coor_pred = self.data_dict['coor_pred'][idx]  # (L, 2)
            inpaint = self.data_dict['inpaint_mask'][idx].reshape(-1, 1)  # (L, 1)
            w, h = self.img_config['img_shape']
            # Normalisation
            coor_pred[:, 0] /= w
            coor_pred[:, 1] /= h
            return data_idx, coor_pred, inpaint
        else:
            raise NotImplementedError





class VideoWindowDataset(Dataset):
    def __init__(self,
                 video_file: str,
                 seq_len: int = 8,
                 sliding_step: int = 1,
                 bg_mode: str = '',
                 HEIGHT: int = HEIGHT,
                 WIDTH: int = WIDTH,
                 max_sample_num: int = 1000,
                 video_range: tuple = None,
                 median: np.ndarray = None):
        self.video_file   = video_file
        self.seq_len      = seq_len
        self.sliding_step = sliding_step
        self.bg_mode      = bg_mode
        self.HEIGHT       = HEIGHT
        self.WIDTH        = WIDTH

        # Ouvre la vidéo pour récupérer longueur et fps
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Liste de tous les indices de début de séquences
        self.starts = list(range(0, total_frames, sliding_step))
        self.video_len = total_frames
        self.fps       = fps

        # Pré-génère la médiane si besoin
        if bg_mode and median is None:
            self.median = self._gen_median(max_sample_num, video_range)
        else:
            self.median = median

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start_f = self.starts[idx]
        cap     = cv2.VideoCapture(self.video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        frames = []
        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # padding avec la dernière image si fin de vidéo
                frames.append(frames[-1].copy())
            else:
                frames.append(frame)
        cap.release()

        # transformation RGB→(C,H,W), concat, bg_mode…
        imgs = np.stack(frames)[..., ::-1]  # BGR→RGB
        processed = self._process(imgs)

        # On transmet aussi les indices de frame pour le post-traitement
        data_idx = [(0, min(start_f + i, self.video_len-1)) for i in range(self.seq_len)]
        data_idx = np.array(data_idx, dtype=np.int64)

        return data_idx, processed

    def _gen_median(self, max_sample_num, video_range):
        print('Generate median image…')
        cap = cv2.VideoCapture(self.video_file)
        total = self.video_len
        if video_range:
            start = min(max(0, video_range[0] * self.fps), total)
            end   = min(video_range[1] * self.fps, total)
        else:
            start, end = 0, total
        seg_len = end - start
        step    = max(1, seg_len // max_sample_num)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        samples = []
        for _ in range(start, end, step):
            ret, f = cap.read()
            if not ret:
                break
            samples.append(f)
            for __ in range(step-1):
                cap.grab()
        cap.release()

        median = np.median(np.stack(samples), axis=0)[..., ::-1]
        if self.bg_mode == 'concat':
            median = cv2.resize(median, (self.WIDTH, self.HEIGHT),
                                interpolation=cv2.INTER_NEAREST)
            median = median.transpose(2,0,1)
        print('Median image generated.')
        return median

    def _process(self, imgs: np.ndarray):
        ch_list = []
        for i in range(self.seq_len):
            resized = cv2.resize(imgs[i], (self.WIDTH, self.HEIGHT),
                                 interpolation=cv2.INTER_LINEAR)
            ch_list.append(resized.transpose(2,0,1))

        stacked = np.concatenate(ch_list, axis=0)  # 24 canaux

        if self.bg_mode == 'subtract':
            diff = cv2.absdiff(stacked[:3], self.median)
            gray = cv2.cvtColor(diff.transpose(1,2,0), cv2.COLOR_RGB2GRAY)[None]
            stacked = np.concatenate((gray, stacked[3:]), axis=0)

        elif self.bg_mode == 'concat':
            stacked = np.concatenate((self.median, stacked), axis=0)

        return (stacked.astype(np.float32) / 255.0)