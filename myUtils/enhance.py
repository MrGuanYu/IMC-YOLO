# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy
from PIL import Image
import shutil


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.6):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.6):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    # flipped_image = np.fliplr(image)
    # return flipped_image
    flip_image = cv2.flip(image,1)
    return flip_image


from PIL import Image, ImageEnhance
import os
import random
import shutil

# target_num = 250  # 目标增强图片数量
image_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\finalGuodu\train'  # 图片文件夹路径
save_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\final\train'  # 保存增强后的图片的文件夹路径


# label_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\mdAAA\train\labels'
# save_label_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasetsOrigin\train\labels'


def augment_image(image):
    image_path = os.path.join(image_folder, 'images')
    save_path = os.path.join(save_folder, 'images')

    img = cv2.imread(os.path.join(image_path, image))
    cv2.imwrite(os.path.join(save_path,image),img)
    shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
                os.path.join(save_folder, 'labels', image.split('.')[0] + '.txt'))

    # cv2.imshow("1",img)
    # cv2.waitKey(5000)
    # 旋转
    # rotated_90 = rotate(img, 90)
    # cv2.imwrite(save_path + "".join(name) + '_r90.' + extension, rotated_90)
    # rotated_180 = rotate(img, 180)
    # cv2.imwrite(save_path + "".join(name) + '_r180.' + extension, rotated_180)
    # flipped_img = flip(img)
    #
    # cv2.imwrite(os.path.join(save_path, image.split('.')[0]+'_flip.jpg'), flipped_img)
    # shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
    #             os.path.join(save_folder, 'labels', image.split('.')[0] + '_flip.txt'))
    # with open(os.path.join(save_folder, 'labels', image.split('.')[0] + '_flip.txt'), 'r') as file:
    #     lines = file.readlines()
    # modified_lines = []
    # for line in lines:
    #     parts = line.split()  # 将每行拆分为单词或数字
    #     if len(parts) >= 2:  # 确保至少有两个元素
    #         try:
    #             number = parts[1]  # 尝试将第二个元素转换为整数
    #             new_number = 1.0 - float(number)  # 更改数字的值
    #             parts[1] = str(new_number)  # 将新值转换为字符串
    #         except ValueError:
    #             pass  # 跳过无法转换为整数的行
    #
    #     modified_line = ' '.join(parts)  # 将更新后的行重新连接起来
    #     modified_lines.append(modified_line)
    # with open(os.path.join(save_folder,'labels', image.split('.')[0] + '_flip.txt'), 'w') as file:
    #     file.writelines(modified_lines)

    # 增加噪声
    # img_salt = SaltAndPepper(img, 0.3)
    # cv2.imwrite(save_path + img_name[0:7] + '_salt.jpg', img_salt)
    # img_gauss = addGaussianNoise(img, 0.5)
    # cv2.imwrite(os.path.join(save_path, image.split('.')[0]+'_gauss.jpg'), img_gauss)
    # shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
    #             os.path.join(save_folder, 'labels', image.split('.')[0] + '_gauss.txt'))
    img_salt = SaltAndPepper(img, 0.5)
    cv2.imwrite(os.path.join(save_path, image.split('.')[0] + '_salt.jpg'), img_salt)
    shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
                os.path.join(save_folder, 'labels', image.split('.')[0] + '_salt.txt'))

    # 变亮、变暗
    img_darker = darker(img)
    cv2.imwrite(os.path.join(save_path, image.split('.')[0]+'_darker.jpg'), img_darker)
    shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
                os.path.join(save_folder, 'labels', image.split('.')[0] + '_darker.txt'))
    img_brighter = brighter(img)
    cv2.imwrite(os.path.join(save_path, image.split('.')[0]+'_brighter.jpg'), img_brighter)
    shutil.copy(os.path.join(image_folder, 'labels', image.split('.')[0] + '.txt'),
                os.path.join(save_folder, 'labels', image.split('.')[0] + '_brighter.txt'))

    # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    # cv2.imwrite(save_path + "".join(name) + '_blur.' + extension, blur)


# 获取所有类别的文件夹路径
imageList = os.listdir(os.path.join(image_folder,'images'))

for n,image in enumerate(imageList):
    sourcePath = os.path.join(image_folder, image)
    targetPath = os.path.join(save_folder)
    # image=np.array(image)
    augment_image(image)
    print("第{}张图片的增强全部完成".format(n + 1))

