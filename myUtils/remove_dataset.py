import os

AIM_DIR = r"D:\program\python\ultralytics_withV9\myDatasets\datasets\ShellNANI\train\\"

OLD_LABEL = AIM_DIR + r"labels\\"
NEW_LABEL = AIM_DIR + r"target_labels\\"
OLD_IMAGE = AIM_DIR + r"images\\"
NEW_IMAGE = AIM_DIR + r"target_images\\"

# 创建新文件夹
if not os.path.exists(NEW_LABEL[:-1]):
    os.makedirs(NEW_LABEL[:-1])
if not os.path.exists(NEW_IMAGE[:-1]):
    os.makedirs(NEW_IMAGE[:-1])

# 获取路径内的所有文件路径列表
yolo_file = os.listdir(OLD_LABEL)

# 遍历文件夹
for label_name in yolo_file:
    # 打开文件
    old_label_path = OLD_LABEL + label_name
    flag = False
    with open(old_label_path, "r+") as f:
        if f.read().strip() == '':
            flag = True
        for line in f:
            # 获取每行数字个数, 如果数字个数大于5
            if len(line.split(' ')) > 5 :
                flag = True
                break

    # 关闭文件之后再进行文件移动，避免冲突
    if flag:
        old_image_path = OLD_IMAGE + label_name[:-3] + "jpg"

        new_image_path = NEW_IMAGE + label_name[:-3] + "jpg"
        new_label_path = NEW_LABEL + label_name

        os.rename(old_label_path, new_label_path)
        os.rename(old_image_path, new_image_path)

        print(new_label_path)