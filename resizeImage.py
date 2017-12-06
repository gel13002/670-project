import os
from PIL import Image
from resizeimage import resizeimage

dir = './Automatically_Annotated_Images'
count = 0
for folder in os.listdir(dir):
    if folder.startswith('.'):
        continue
    folderPath = dir + '/' + folder
    for file in os.listdir(folderPath):
        filePath = folderPath + '/' + file
        with open(filePath, 'rb') as f:
            with Image.open(f) as im:
                newimg = resizeimage.resize_cover(im, [128, 128])
                newimg.save('./resized_images_128/' + file, im.format)