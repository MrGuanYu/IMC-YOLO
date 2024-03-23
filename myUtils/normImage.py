import os
import cv2

# sourcePath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400hole_beifen\val\images'
# targetPath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400holeGuodu\val\images'

#
# imageList = os.listdir(sourcePath)
#
# for img in imageList:
#     print(os.path.join(sourcePath, img))
#     image = cv2.imread(os.path.join(sourcePath, img))
#     height, width,_ = image.shape
#     if height > width:
#         rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#         cv2.imwrite(os.path.join(targetPath, img), rotated_img)
#     else:
#         cv2.imwrite(os.path.join(targetPath, img), image)


sourcePath = r'D:\program\python\ultralytics_withV9\myDatasets\datasets\400holeGuodu\val\images'

imageList = os.listdir(sourcePath)

for img in imageList:
    image = cv2.imread(os.path.join(sourcePath, img))
    image = cv2.resize(image,(1080,960))
    cv2.imwrite(os.path.join(sourcePath,img),image)





# 960 * 1080
