import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics/cfg/models/v8/IMC-YOLO.yaml')
    model.train(task='detect',
                data=r'dataset/razor_clam_burrows/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=100,
                workers=0,
                device='0',
                optimizer='Adam',
                amp=False,
                name='IMC-YOLO',
                pretrained=False,
                patience=100,
                lr0=0.001,
                plots=True,
                project='runs'
                )
