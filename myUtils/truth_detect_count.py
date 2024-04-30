import os

from PIL import Image, ImageDraw, ImageFont


def parse_yolo_annotation(annotation_file):
    boxes = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 将每行的字符串按空格分割，并转换为浮点数
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            # 将预测框的信息添加到列表中
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes





def draw_boxes(image_path, boxes):
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 遍历每个预测框
    num = 0
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # 计算预测框的四个角的坐标
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)

        num += 1
        # 绘制预测框
        draw.rectangle([x1, y1, x2, y2], outline="red",width=4)

    # 确定字体大小（以像素为单位）
    font_size = 27  # 可以根据需要调整字体大小

    # 加载字体
    font = ImageFont.truetype("arial.ttf", font_size)

    # 在左上角添加计数器
    draw.text((10, 5), f"Total Holes:{len(boxes)}", fill=(0, 0, 255), font=font)

    tempPath = r'D:\program\python\ultralytics_withV9\myRubbish\4_27'
    output_path = os.path.join(r'D:\program\python\ultralytics_withV9\myRubbish\4_27',"truth_" + image_path.split('\\')[-1])
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)

    image.save(output_path)

if __name__ == '__main__':

    # imagePath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\train\images'
    # labelPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\train\labels'
    # labelList = os.listdir(labelPath)
    # for label in labelList:
    #     name = label.split('.')[0]
    #     annotation_file = os.path.join(labelPath,name + '.txt')
    #     image_path = os.path.join(imagePath,name + '.jpg')
    #     boxes = parse_yolo_annotation(annotation_file)
    #     draw_boxes(image_path, boxes)


    annotation_file = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\test\labels\IMG_20240310_145400_bottom_flip.txt'
    image_path = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix\test\images\IMG_20240310_145400_bottom_flip.jpg'
    boxes = parse_yolo_annotation(annotation_file)
    draw_boxes(image_path,boxes)