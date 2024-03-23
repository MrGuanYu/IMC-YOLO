import xml.etree.ElementTree as ET
import os
import argparse
import random


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def mkDataset(imageName_list, voc_folder, yolo_folder,
              datasetType, class_name):
    i = 0
    for imageName in imageName_list:
        name = imageName.split('.')[0]
        print(name)
        imagefilePath = os.path.join(voc_folder, imageName)  # 图片源路径
        xmlfilePath = os.path.join(voc_folder, name + '.xml')  # xml源路径
        print(xmlfilePath)
        # xmlfilePath.encode('utf-8')

        os.makedirs(os.path.join(yolo_folder, datasetType, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(yolo_folder, datasetType, 'images'), exist_ok=True)

        if os.path.exists(xmlfilePath):
            print("xml存在")
            with open(xmlfilePath, 'rb+') as in_file:
                print(in_file)
                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find('size')

                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    cls_id = class_name.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text),
                         float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text),
                         float(xmlbox.find('ymax').text))
                    box = convert((w, h), b)

                    # 写入yolo的txt标注格式文件->myDatasets/Datasets/xxxdataset/train(举例)/labels
                    with open(os.path.join(yolo_folder, datasetType, 'labels', name + '.txt'), 'a') as out_file:
                        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in box]) + '\n')
                        print("good")

                    # 读取原图片
                    with open(os.path.join(imagefilePath), 'rb') as image_file:
                        content = image_file.read()

                    # 写入原图片
                    with open(os.path.join(yolo_folder, datasetType, 'images', imageName), 'wb') as out_file:
                        out_file.write(content)
        else:
            print("{}份xml文件出错".format(i))
            i += 1


def voc2yolo(voc_folder, yolo_folder, class_name, trainRatio, valRatio, testRatio):
    fileList = os.listdir(voc_folder)  # xx1.jpg,xx1,xml,xx2.png,xx2.xml~~~
    imageName_list = []  # xx1.jpg,xx2.png,xx3.JPG
    for file in fileList:
        # 若文件名包含多个'.'，此处代码要更改
        imageName_list.append(file) if file.endswith(('.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG')) else None

    name_list_len = len(imageName_list)
    random.shuffle(imageName_list)
    print(imageName_list)

    datasetTypeDict = {'train': trainRatio,
                       'val': valRatio,
                       'train': testRatio}
    index = 0
    for datasetType, ratio in datasetTypeDict.items():
        mkDataset(imageName_list=imageName_list[index:index + int(ratio * name_list_len)],
                  voc_folder=voc_folder,
                  yolo_folder=yolo_folder,
                  datasetType=datasetType,
                  class_name=class_name)
        index += int(ratio * name_list_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_folder', type=str)  # myDatasets/originDatasets/xxxdataset
    parser.add_argument('--yolo_folder', type=str)  # myDatasets/Datasets/xxxdataset
    parser.add_argument('--class_name', type=str)  # "dog,cat,pig,sheep"
    parser.add_argument('--trainRatio', default=0.7, type=float)
    parser.add_argument('--valRatio', default=0.15, type=float)
    parser.add_argument('--testRatio', default=0.15, type=float)
    args = parser.parse_args()

    voc_folder = args.voc_folder
    yolo_folder = args.yolo_folder
    class_name = args.class_name.split(',')
    trainRatio = args.trainRatio
    valRatio = args.valRatio
    testRatio = args.testRatio

    voc2yolo(voc_folder=voc_folder, yolo_folder=yolo_folder, class_name=class_name,
             trainRatio=trainRatio, valRatio=valRatio, testRatio=testRatio)
