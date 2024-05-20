import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR
# unfold 1/2 1/2 hwd 四图拼一图
if __name__ == '__main__':

    model4 = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model4.train(task='detect',
                data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\ShellNANI\data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=200,
                workers=0,
                device='0',
                optimizer='Adam', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                # project='runs/train
                name='yolov8n_ShellNANI',
                pretrained=False,
                patience=150,
                lr0=0.001,
                plots=True,
                single_cls=True,
                # weight_decay=0.0005
                )


    model3 = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-head-tcambQ_myaifi-fasterblock.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model3.train(task='detect',
                data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\ShellNANI\data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=200,
                workers=0,
                device='0',
                optimizer='Adam', # using SGD
                # resume='', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                # project='runs/train
                name='yolov8n_mine_ShellNANI',
                pretrained=False,
                patience=150,
                lr0=0.001,
                plots=True,
                single_cls=True,

                # weight_decay=0.0005
                )


