import os
import re


def remove_chinese_in_filenames(directory):
    # 遍历指定目录下的所有文件名
    for filename in os.listdir(directory):
        # 构造完整的文件路径
        filepath = os.path.join(directory, filename)

        # 判断是否为文件
        if os.path.isfile(filepath):
            # 使用正则表达式匹配中文字符
            chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')

            # 将中文字符替换为空字符串
            new_filename = re.sub(chinese_pattern, '', filename)

            # 构造新的文件路径
            new_filepath = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(filepath, new_filepath)


# 指定需要删除中文的文件所在目录
directory = r'D:\program6\python\ultralytics\myDatasets\originDatasets\blood_cell'

# 调用函数删除中文文件名
remove_chinese_in_filenames(directory)