import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    model = YOLO(r'ultralytics/cfg/models/v8/yolov8n-head-mycbam-aifi.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(task='detect',
                data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=100,
                workers=0,
                device='0',
                optimizer='Adam', # using SGD

                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                # project='runs/train',
                name='yolov8n-head-mycbam-aifi-innersiou',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                # weight_decay=0.0005
                )