import os

import cv2


def is_image_blurry(image_path, threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    print(fm)

    if fm < threshold:
        return True  # 模糊
    else:
        return False  # 非模糊


def del_blurry_all(filelist_path, threshold_value):
    file_list = os.listdir(filelist_path)

    origin_nums = len(file_list)

    for file in file_list:
        file_path = os.path.join(filelist_path, file)
        if file.endswith(
                ('.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG')):
            is_blurry = is_image_blurry(file_path, threshold_value)
            fileName = file.split('.')[0]
            label_path = os.path.join(filelist_path, fileName + '.json')
            if not os.path.exists(label_path):
                os.remove(file_path)
            elif is_blurry:
                os.remove(file_path)
                os.remove(label_path)
            else:
                pass


    now_nums = len(os.listdir(filelist_path))

    return 1. - now_nums / origin_nums


if __name__ == "__main__":
    # 设定清晰度阈值
    threshold_value = 35.0

    # 标注文件夹路径
    filelist_path = r"D:\others6\work\temp_bingo_labeling_1"

    # 判断文件夹内图片是否模糊，糊则删image和label
    ratio = del_blurry_all(filelist_path, threshold_value)

    print("本次删除了{:.2f}%".format(ratio * 100))
