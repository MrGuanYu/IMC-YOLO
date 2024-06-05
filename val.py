import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

import matplotlib.pyplot as plt
plt.switch_backend('agg')

if __name__ == '__main__':
    model = YOLO(r'runs/IMC-YOLO/weights/best.pt')
    metrics = model.val(data=r'dataset/razor_clam_burrows/data.yaml',
                        split='test',
                        cache=False,
                        save_json=True,  # if you need to cal coco metrice
                        project='runs/val',
                        name='exp',
                        plots=True
                        )
