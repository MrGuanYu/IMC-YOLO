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
def darker(image, percetage=0.9):
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
def brighter(image, percetage=1.5):
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
    flipped_image = np.fliplr(image)
    return flipped_image


from PIL import Image, ImageEnhance
import os
import random
import shutil


def augment_image(source_path, target_path):

    image_path = os.path.join(source_path,'images')
    save_path = os.path.join(target_path,'images')
    label_path = os.path.join(source_path,'labels')
    label_save_path = os.path.join(target_path,'labels')

    img = cv2.imread(image_path)
    image_name = os.path.basename(image_path)  # 获取图片名称
    split_result = image_name.split('.')
    name = split_result[:-1]
    extension = split_result[-1]
    # cv2.imshow("1",img)
    # cv2.waitKey(5000)
    # 旋转
    # rotated_90 = rotate(img, 90)
    # cv2.imwrite(save_path + "".join(name) + '_r90.' + extension, rotated_90)
    # rotated_180 = rotate(img, 180)
    # cv2.imwrite(save_path + "".join(name) + '_r180.' + extension, rotated_180)
    flipped_img = flip(img)
    cv2.imwrite(save_path + "".join(name) + '_fli.' + extension, flipped_img)


    # 增加噪声
    # img_salt = SaltAndPepper(img, 0.3)
    # cv2.imwrite(save_path + img_name[0:7] + '_salt.jpg', img_salt)
    img_gauss = addGaussianNoise(img, 0.3)
    cv2.imwrite(save_path + "".join(name) + '_noise.' + extension, img_gauss)
    shutil.copy(os.path.join(label_path,))

    # 变亮、变暗
    img_darker = darker(img)
    cv2.imwrite(save_path + "".join(name) + '_darker.' + extension, img_darker)
    img_brighter = brighter(img)
    cv2.imwrite(save_path + "".join(name) + '_brighter.' + extension, img_brighter)

    # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    # cv2.imwrite(save_path + "".join(name) + '_blur.' + extension, blur)

# target_num = 250  # 目标增强图片数量
image_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400_split_withoutHence\test'  # 图片文件夹路径
save_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400to1600_final\test'  # 保存增强后的图片的文件夹路径
# label_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\mdAAA\train\labels'
# save_label_folder = r'D:\program\python\ultralytics_withV9\myDatasets\datasetsOrigin\train\labels'

# 获取所有类别的文件夹路径
class_folders = os.listdir(image_folder)

# 遍历类别文件夹
for class_folder in class_folders:
    if not os.path.isdir(os.path.join(image_folder, class_folder)):
        continue
    target_subfolder = os.path.join(save_folder, class_folder)
    os.makedirs(target_subfolder, exist_ok=True)
    image_list = os.listdir(os.path.join(image_folder, class_folder))
    # 获取当前文件夹中所有图片的路径
    images = []
    for file_name in image_list:
        images.append(os.path.join(image_folder, class_folder, file_name))
    num_images = len(images)
    print(num_images)
    print(target_num)
    if num_images < target_num:
        for image_path in images:
            with Image.open(image_path) as img:
                name = os.path.basename(image_path)
                target_path = os.path.join(target_subfolder, name)
                shutil.copy(image_path, target_path)
        i = num_images
        j = 0
        random_image = random.sample(image_list, k=num_images)
        while i < target_num and j <= num_images - 1:
            image_path = os.path.join(image_folder, class_folder, random_image[j])
            target_path = target_subfolder + '/'
            augment_image(image_path, target_path)
            i += 7
            j += 1
            print(i)
    else:
        # 随机选择2000张图片
        selected_images = random.sample(images, k=2000)
        # 将选中的图片复制到目标文件夹
        for image_path in selected_images:
            with Image.open(image_path) as img:
                name = os.path.basename(image_path)
                target_path = os.path.join(target_subfolder, name)
                shutil.copy(image_path, target_path)