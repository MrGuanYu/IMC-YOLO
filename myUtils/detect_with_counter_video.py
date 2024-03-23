# from ultralytics.utils import ASSETS
# from ultralytics.models.yolo.detect import DetectionPredictor
#
# args = dict(model=r'D:\program6\python\ultralytics\runs\detect\train5\weights\best.pt', source=r'D:\program6\python\ultralytics\myDatasets\Datasets\testVideo.mp4')
# predictor = DetectionPredictor(overrides=args)
#
# predictor.predict_cli()

import cv2
from ultralytics import YOLO


# Load the YOLOv8 model
model = YOLO(r'D:\program6\python\ultralytics\runs\detect\train5\weights\best.pt')

# Open the video file
video_path = r"D:\program6\python\ultralytics\myDatasets\Datasets\testVideo.mp4"
cap = cv2.VideoCapture(video_path)

print(cap.isOpened())

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output video file path
output_path = r"D:\program6\python\ultralytics\myDatasets\Datasets\video.mp4"

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize a counter to keep track of the total number of prediction boxes
total_boxes = 0

# Initialize a dictionary to keep track of the count for each class
class_count = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        print("\n\n\n\n\n\n\n" + "results:")
        print(results)
        print("\n\n\n")
        print(results[0].names)
        print("\n\n\n")

        print("results[0]:")
        print(results[0])
        print("\n\n\n")

        # Get the number of prediction boxes in the current frame
        total_boxes = len(results[0].boxes)
        print("results[0].boxes:")
        print(results[0].boxes)
        print("\n\n\n")

        print("results[0].boxes.cls:")
        print(results[0].boxes.cls)
        print("\n\n\n")

        clsNum = len(results[0].names)
        cls = {}
        for i in range(clsNum):
            cls[i] = 0

        for v in results[0].boxes.cls:
            for key,value in cls.items():
                if v ==  key:
                    value += 1
                    cls[key] = value

        # for box in results[0].boxes:
        #     cls[str(box[5].item())] = 1 if box[5] not in cls else cls[str(box[5].item())] += 1


        print("cls:  " + str(cls))

        # 选择结果是否包含置信度信息
        annotated_frame = results[0].plot(conf=False)
        # annotated_frame = results[0].plot()

        # Display the annotated frame with the total number of prediction boxes
        cv2.putText(annotated_frame, f'Total Boxes: {total_boxes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for key,value in cls.items():
            pixes = 60

            name = results[0].names[key]

            cv2.putText(annotated_frame,
                         '{} Num:{}'.format(name,str(value)),
                        (10, pixes),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            pixes += 30
        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    # cv2.waitKey(1000)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
