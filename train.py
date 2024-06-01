import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR
# unfold 1/2 1/2 hwd 四图拼一图
if __name__ == '__main__':
    model = YOLO(r'D:\program\python\IMC-YOLO\ultralytics\cfg\models\v8\IMC-YOLO.yaml')
    # model = RTDETR(r'D:\program\python\ultralytics_withV9\ultralytics\cfg\models\rt-detr\rtdetr-l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(task='detect',
                data=r'D:\program\python\IMC-YOLO\dataset\razor_clam_burrows\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=100,
                workers=0,
                device='0',
                optimizer='Adam',
                amp=False,
                name='yolov8n',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                project=r'runs'
                )