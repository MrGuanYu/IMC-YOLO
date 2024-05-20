import os
import random
import shutil

originPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\mudflatFinal\temp'
imageList = os.listdir(os.path.join(originPath, 'images'))
targetPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\mudflatFinal'

random.shuffle(imageList)

# 默认 7：1.5：1.5
index = 0
# for _ in range(int(len(imageList) * 0.7)):
#     shutil.copy(os.path.join(originPath, 'images', imageList[index]),
#                 os.path.join(targetPath, 'temp', 'images', imageList[index]))
#     shutil.copy(os.path.join(originPath, 'labels', imageList[index].rsplit('.',1)[0] + '.txt'),
#                 os.path.join(targetPath, 'temp', 'labels', imageList[index].rsplit('.',1)[0] + '.txt'))
#     index += 1

for _ in range(int(len(imageList) * 0.5)):
    shutil.copy(os.path.join(originPath, 'images', imageList[index]),
                os.path.join(targetPath, 'test', 'images', imageList[index]))
    shutil.copy(os.path.join(originPath, 'labels', imageList[index].rsplit('.',1)[0] + '.txt'),
                os.path.join(targetPath, 'test', 'labels', imageList[index].rsplit('.',1)[0] + '.txt'))
    index += 1

for _ in range(len(imageList)-index):
    shutil.copy(os.path.join(originPath, 'images', imageList[index]),
                os.path.join(targetPath, 'val', 'images', imageList[index]))
    shutil.copy(os.path.join(originPath, 'labels', imageList[index].rsplit('.',1)[0] + '.txt'),
                os.path.join(targetPath, 'val', 'labels', imageList[index].rsplit('.',1)[0] + '.txt'))
    index += 1