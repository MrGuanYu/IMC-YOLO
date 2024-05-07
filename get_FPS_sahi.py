import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import math
import os
import random
import time
import sys
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = r"D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-tcamb-myaifi\weights\best.pt"
# download_yolov8s_model(yolov8_model_path)
#
# # Download test images
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-tcamb-myaifi\weights\best.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=30, type=int, help='warmup time')
    parser.add_argument('--testtime', default=1000, type=int, help='test time')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()
    
    device = select_device(opt.device, batch=opt.batch)
    
    # Model
    weights = opt.weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = attempt_load_weights(weights, device=device, fuse=True)
        print(f'Loaded {weights}')  # report
    else:
        assert weights.endswith('.pt'), "compress need weights."
    
    # model = model.to(device)
    model = detection_model

    # model.fuse()
    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)
    
    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()
    
    print('begin warmup...')
    for i in tqdm(range(opt.warmup), desc='warmup....'):
        # model(example_inputs)
        for img in example_inputs:
            result = get_sliced_prediction(
                # "demo_data/small-vehicles1.jpeg",
                img.numpy(),
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

    print('begin test latency...')
    time_arr = []
    
    for i in tqdm(range(opt.testtime), desc='test latency....'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        model(example_inputs)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        time_arr.append(end_time - start_time)
    
    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)
    
    print(f'model weights:{opt.weights} size:{get_weight_size(opt.weights)}M (bs:{opt.batch})Latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}')