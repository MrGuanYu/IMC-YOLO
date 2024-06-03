import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\program\python\IMC-YOLO\ultralytics\cfg\models\v8\yolov8n.yaml')
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
                project='runs'
                )
