import os
import random
import shutil

sourcePath = r'D:\program\python\ultralytics_withV9\myDatasets\datasetsOrigin\temp'
targetPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhence_mix'

imageList = os.listdir(os.path.join(sourcePath, 'images'))
random.shuffle(imageList)

index = 0
for _ in range(int(len(imageList) * 0.5)):
    shutil.copy(os.path.join(sourcePath, 'images', imageList[index]),
                os.path.join(targetPath, 'test', 'images', imageList[index]))
    shutil.copy(os.path.join(sourcePath, 'labels', imageList[index].split('.')[0] + '.txt'),
                os.path.join(targetPath, 'test', 'labels', imageList[index].split('.')[0] + '.txt'))
    index += 1

for _ in range(len(imageList) - index):
    shutil.copy(os.path.join(sourcePath, 'images', imageList[index]),
                os.path.join(targetPath, 'val', 'images', imageList[index]))
    shutil.copy(os.path.join(sourcePath, 'labels', imageList[index].split('.')[0] + '.txt'),
                os.path.join(targetPath, 'val', 'labels', imageList[index].split('.')[0] + '.txt'))
    index += 1
