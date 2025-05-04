import os
import cv2
import math
import parse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from collections import deque
from utils.general import HEIGHT, WIDTH, SIGMA, IMG_FORMAT
from line_profiler import LineProfiler 
import numpy as np
import time
from PIL import Image
import torch
data_dir = 'data'



# dataset.py (ou dans predict.py si vous préférez)
from collections import deque
import cv2
import numpy as np
from torch.utils.data import IterableDataset

class CircularVideoDataset(IterableDataset):
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
        self.step         = sliding_step
        self.bg_mode      = bg_mode
        self.HEIGHT       = HEIGHT
        self.WIDTH        = WIDTH

        # Ouvre la vidéo pour longueur & fps
        cap = cv2.VideoCapture(video_file)
        self.video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps       = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Génération de la médiane si mode concat ou subtract
        if bg_mode and median is None:
            self.median = self._gen_median(max_sample_num, video_range)
        else:
            self.median = median

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_file)
        buf = deque(maxlen=self.seq_len)

        # Pré-remplissage du buffer
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break
            buf.append(frame)
        # Pad si vidéo trop courte
        while len(buf) < self.seq_len:
            buf.append(buf[-1].copy())

        start_idx = 0
        total = self.video_len

        while True:
            # 1) Prépare la fenêtre d’images
            window = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buf]
            # 2) Resize + format
            imgs = np.stack([
                cv2.resize(img, (self.WIDTH, self.HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
                for img in window
            ])
            processed = self._process(imgs)

            # 3) Indices pour post-traitement
            idxs = np.clip(start_idx + np.arange(self.seq_len), 0, total-1)
            data_idx = np.stack([idxs, idxs], axis=1).astype(np.int64)
            yield data_idx, processed

            # 4) Glissement du buffer
            ret, next_frame = cap.read()
            if not ret:
                break
            buf.append(next_frame)
            start_idx += self.step

        cap.release()

    def _gen_median(self, max_sample_num, video_range, downsample=4):
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

        # Tailles réduites
        small_w = max(1, self.WIDTH  // downsample)
        small_h = max(1, self.HEIGHT // downsample)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        samples = []
        for _ in range(start, end, step):
            ret, f = cap.read()
            if not ret:
                break
            # downscale pour accélérer la médiane
            f_small = cv2.resize(f, (small_w, small_h),
                                interpolation=cv2.INTER_AREA)
            samples.append(f_small)
            for __ in range(step-1):
                cap.grab()
        cap.release()

        # médiane sur le petit volume (N, small_h, small_w, 3)
        median_small = np.median(np.stack(samples), axis=0)[..., ::-1]  # BGR→RGB
        # upscale en pleine résolution
        median = cv2.resize(median_small, (self.WIDTH, self.HEIGHT),
                            interpolation=cv2.INTER_NEAREST)
        # transpose pour concat mode
        median = median.transpose(2,0,1)  # (3, H, W)
        print('Median image generated.')
        return median

    def _process(self, imgs: np.ndarray):
        # imgs = (seq_len, H, W, 3) RGB
        ch_list = [imgs[i].transpose(2,0,1) for i in range(self.seq_len)]
        stacked = np.concatenate(ch_list, axis=0)  # (3*seq_len, H, W)


        if self.bg_mode == 'concat':
            stacked = np.concatenate((self.median, stacked), axis=0)

        return (stacked.astype(np.float32) / 255.0)
