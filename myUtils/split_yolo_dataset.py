import os
import random
import shutil

originPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasetsOrigin\hunhe_hole'
imageList = os.listdir(os.path.join(originPath, 'images'))
targetPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400hole'

random.shuffle(imageList)

# 默认 8：1：1
index = 0
for _ in range(int(len(imageList) * 0.7)):
    shutil.copy(os.path.join(originPath, 'images', imageList[index]),
                os.path.join(targetPath, 'train', 'images', imageList[index]))
    shutil.copy(os.path.join(originPath, 'labels', imageList[index].split('.')[0] + '.txt'),
                os.path.join(targetPath, 'train', 'labels', imageList[index].split('.')[0] + '.txt'))
    index += 1

for _ in range(int(len(imageList) * 0.15)):
    shutil.copy(os.path.join(originPath, 'images', imageList[index]),
                os.path.join(targetPath, 'test', 'images', imageList[index]))
    shutil.copy(os.path.join(originPath, 'labels', imageList[index].split('.')[0] + '.txt'),
                os.path.join(targetPath, 'test', 'labels', imageList[index].split('.')[0] + '.txt'))
    index += 1

for _ in range(len(imageList)-index):
    shutil.copy(os.path.join(originPath, 'images', imageList[index]),
                os.path.join(targetPath, 'val', 'images', imageList[index]))
    shutil.copy(os.path.join(originPath, 'labels', imageList[index].split('.')[0] + '.txt'),
                os.path.join(targetPath, 'val', 'labels', imageList[index].split('.')[0] + '.txt'))
    index += 1