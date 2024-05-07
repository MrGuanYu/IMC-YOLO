from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = r"runs/detect/yolov8n-head-tcamb-myaifi/weights/best.pt"


# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov8',
#     model_path=yolov8_model_path,
#     confidence_threshold=0.3,
#     device="cuda:0",  # or 'cuda:0'
# )

predict(
    model_type="yolov8",
    model_path=yolov8_model_path,
    model_device="cuda:0",  # or 'cuda:0'
    model_confidence_threshold=0.3,
    source=r"D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\test\images",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    # model_category_mapping={"0":'hole'},
    # model_category_remapping={"0":'hole'},
    # postprocess_class_agnostic=True,
    dataset_json_path=r'D:\program\python\ultralytics_withV9\myRubbish\annotations\instances_test.json',
    no_standard_prediction=True

)