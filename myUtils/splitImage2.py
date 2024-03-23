from PIL import Image
import os

resourcePath = r'../myDatasets/datasetsOrigin/teachertemp1'
targetPath = r'../myDatasets/datasetsOrigin/tempHole'

imageList = os.listdir(resourcePath)

for img in imageList:
    image = Image.open(os.path.join(resourcePath, img))

    width, height = image.size

    if width < height:
        topImage = image.crop((0, 0, width, height // 2))
        bottomImage = image.crop((0, height // 2, width, height))

        topImage.save(os.path.join(targetPath, img.split('.')[0] + '_top.' + img.split('.')[1]))
        bottomImage.save(os.path.join(targetPath, img.split('.')[0] + '_bottom.' + img.split('.')[1]))
    else:
        leftImage = image.crop((0, 0, width//2, height))
        rightImage = image.crop((width//2, 0, width, height))

        leftImage.save(os.path.join(targetPath, img.split('.')[0] + '_left.' + img.split('.')[1]))
        rightImage.save(os.path.join(targetPath, img.split('.')[0] + '_right.' + img.split('.')[1]))

