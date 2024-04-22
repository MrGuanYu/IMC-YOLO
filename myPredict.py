from ultralytics import YOLO
from PIL import Image
import cv2


# Load a pretrained YOLOv8n model
model = YOLO(r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-mycbam-aifi2\weights\best.pt')

# Run inference on an image
# results = model('bus.jpg')  # list of 1 Results object
results = model(r'D:\program\python\ultralytics_withV9\myDatasets\datasets\hole\test\images\2_6_2.jpg')  # list of 2 Results objects
# for rlt in results:
for rlt in results:
    rlt.show()

