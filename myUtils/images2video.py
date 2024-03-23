import cv2
import os

if __name__ == "__main__":

    # 使用函数
    image_folder = r'D:\program6\python\ultralytics\myDatasets\Datasets\2024_2_8\test\images'
    output_video_file = r'D:\program6\python\ultralytics\myDatasets\Datasets\testVideo2.mp4'
    fps = 5  # 或者根据需要调整帧率

    # 修改文件名使得包含顺序信息
    # for imageName in os.listdir(image_folder):
    #     name = imageName.split('_')[1]
    #     oldFile = os.path.join(image_folder,imageName)
    #     newFile = os.path.join(image_folder,name)
    #     os.rename(oldFile,newFile)


    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split(".")[0]))  # 按照文件名的数字部分排序

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    print("视频生成完成")