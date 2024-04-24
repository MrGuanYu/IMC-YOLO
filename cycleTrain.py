import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR
# unfold 1/2 1/2 hwd 四图拼一图
if __name__ == '__main__':
    # model = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi.yaml')
    # # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    # model.train(task='detect',
    #             data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\data.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             batch=16,
    #             close_mosaic=100,
    #             workers=0,
    #             device='0',
    #             optimizer='Adam',  # using SGD
    #             # resume='', # last.pt path
    #             amp=False,  # close amp
    #             # fraction=0.2,
    #             # project='runs/train',
    #             name='yolov8n-myhead-myaifi',
    #             pretrained=False,
    #             patience=100,
    #             lr0=0.001,
    #             plots=True,
    #             # weight_decay=0.0005
    #             )

    # model1 = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi-TBCAM1.yaml')
    # # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    # model1.train(task='detect',
    #             data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\data.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             batch=16,
    #             close_mosaic=100,
    #             workers=0,
    #             device='0',
    #             optimizer='Adam', # using SGD
    #             # resume='', # last.pt path
    #             amp=False, # close amp
    #             # fraction=0.2,
    #             # project='runs/train',
    #             name='yolov8n-myhead-myaifi-TBCAM1',
    #             pretrained=False,
    #             patience=100,
    #             lr0=0.001,
    #             plots=True,
    #             # weight_decay=0.0005
    #             )

    model2 = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi-TBCAM2.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model2.train(task='detect',
                data=r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=100,
                workers=0,
                device='0',
                optimizer='Adam',  # using SGD
                # resume='', # last.pt path
                amp=False,  # close amp
                # fraction=0.2,
                # project='runs/train',
                name='yolov8n-myhead-myaifi-TBCAM2',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                # weight_decay=0.0005
                )

    model3 = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi-TBCAM3.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model3.train(task='detect',
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
                name='yolov8n-myhead-myaifi-TBCAM3',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                # weight_decay=0.0005
                )

    model = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi-TBCAM4.yaml')
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
                name='yolov8n-myhead-myaifi-TBCAM4',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                # weight_decay=0.0005
                )

    model = YOLO(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\v8\yolov8n-myhead-myaifi-TBCAM5.yaml')
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
                name='yolov8n-myhead-myaifi-TBCAM5',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                # weight_decay=0.0005
                )