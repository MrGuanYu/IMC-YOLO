import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import matplotlib.pyplot as plt
plt.switch_backend('agg')
# 这里～～～～～～～～～ ：（
if __name__ == '__main__':
    model = YOLO(r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-mycbam-myaifi\weights\best.pt')
    metrics = model.val(data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\data.yaml',
              split = 'test',
              cache=False,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              plots=True
              )

    rlt = metrics.box.map75  # map75
    print(rlt)


