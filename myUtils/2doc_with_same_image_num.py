import os
from PIL import Image
from collections import Counter

def get_image_files(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def compare_images(file1, file2):
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    return image1.tobytes() == image2.tobytes()

def count_duplicate_images(directory1, directory2):
    image_files1 = get_image_files(directory1)
    image_files2 = get_image_files(directory2)

    image_hashes1 = [hash(Image.open(file).tobytes()) for file in image_files1]
    image_hashes2 = [hash(Image.open(file).tobytes()) for file in image_files2]

    duplicates = Counter(image_hashes1) & Counter(image_hashes2)
    return sum(duplicates.values())

# 指定两个文件夹路径进行比较
directory1 = r"D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhance_testInVal\test\images"
directory2 = r"D:\program\python\ultralytics_withV9\myDatasets\datasets\700hole_enhance_testInVal\test\images"

duplicate_count = count_duplicate_images(directory1, directory2)
print("Number of duplicate images:", duplicate_count)