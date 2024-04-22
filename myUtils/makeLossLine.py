import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
# 设置为TkAgg或Qt5Agg
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

import matplotlib.pyplot as plt

# ... 绘图操作 ...

plt.show()



baselineRlt = r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n\results.csv'
improveRlt = r'D:\program\python\ultralytics_withV9\runs\detect\yolov8n-head-mycbam-aifi\results.csv'


# 读取第一个 CSV 文件
df1 = pd.read_csv(baselineRlt)
# 读取第二个 CSV 文件
df2 = pd.read_csv(improveRlt)

column_names = df1.columns
print(column_names)

# 提取每个文件中的 loss 列
loss1 = df1["       metrics/mAP50(B)"]
loss2 = df2["       metrics/mAP50(B)"]

# 创建一个新的图形
plt.figure()

# 绘制第一个曲线
plt.plot(loss1, label="File 1")

# 绘制第二个曲线
plt.plot(loss2, label="File 2")

# 添加标题和标签
plt.title("mAP")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 添加图例
plt.legend()

# 显示图形
plt.show()