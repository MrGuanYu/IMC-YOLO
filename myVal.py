import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__ == '__main__':
    model = YOLO(r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-myhwd2\weights\best.pt')
    model.val(data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhance_03\data.yaml',
              split = 'test',
              cache=False,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )


