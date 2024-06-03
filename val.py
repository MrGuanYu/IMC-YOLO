import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__ == '__main__':
    model = YOLO(r'D:\program\python\IMC-YOLO\runs\yolov8n\weights\best.pt')
    metrics = model.val(data=r'D:\program\python\IMC-YOLO\dataset\razor_clam_burrows\data.yaml',
                        split='test',
                        cache=False,
                        save_json=True,  # if you need to cal coco metrice
                        project='runs/val',
                        name='exp',
                        plots=True
                        )
