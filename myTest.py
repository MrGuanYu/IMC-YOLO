from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'runs/detect/yolov8n/weights/best.pt')

# Define path to the image file
source = r'myDatasets/datasets/700hole_enhence_mix/test/images'

# Run inference on the source
results = model.predict(source,
                        iou=0.6,
                        conf=0.001,
                        imgsz=640,
                        device=0,
                        save_txt=True
                        )  # list of Results objects