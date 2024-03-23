import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__ == '__main__':
    model = YOLO(r'D:\program\python\ultralytics_withV9\runs\detect\yolov8s-p2\weights\best.pt')
    model.val(data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\finall!!!!!_Test\data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )