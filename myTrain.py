import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v5/yolov5s.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(task='detect',
                data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\finall!!!!!_Test\data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                # close_mosaic=10,
                workers=0,
                device='0',
                optimizer='Adam', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                # project='runs/train',
                name='yolov5s',
                pretrained=False,
                patience=100,
                lr0=0.001,
                amp=False,
                plots=True,
                weight_decay=0.0005
                )