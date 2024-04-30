import os.path
import cv2
from ultralytics import YOLO


def counter_object_num_image(model, image_path, output_path):
    image_name = os.path.basename(image_path)

    image = cv2.imread(image_path)

    results = model(image)

    total_boxes = len(results[0].boxes)

    clsNum = len(results[0].names)
    cls = {}
    for i in range(clsNum):
        cls[i] = 0

    for v in results[0].boxes.cls:
        for key, value in cls.items():
            if v == key:
                value += 1
                cls[key] = value

    # 选择结果是否包含置信度信息
    annotated_img = results[0].plot(conf=True)
    # annotated_img = results[0].plot()

    # cv2.putText(annotated_img, f'Total Number: {total_boxes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for key, value in cls.items():
        pixes = 60

        name = results[0].names[key]

        cv2.putText(annotated_img,
                    # '{} Number:{}'.format(name, str(value))
                    f"Total Holes:{str(value)}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)
        pixes += 30

    # cv2.imwrite(os.path.join(output_path, image_name), annotated_img)
    cv2.imwrite(os.path.join(output_path, 'v8n_me6_' + image_name,), annotated_img)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_img)


if __name__ == '__main__':
    # Load the YOLOv8 model
    model = YOLO(r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-tcamb-myaifi6\weights\best.pt')
    image_path = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\test\images\IMG_20240310_145400_bottom_flip.jpg'
    output_path = r'D:\program\python\ultralytics_withV9\myRubbish\4_27'

    counter_object_num_image(model=model, image_path=image_path, output_path=output_path)
