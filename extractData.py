"""
    extract data from 128*128 images
"""

import csv
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def buildDir(csvfile):
    """
    build lookup directory for input csv file
    :param csvfile: name of csv file
    :type csvfile: str
    :return: a directory maps image name to type
    """
    image_dir = dict()
    with open(csvfile) as file:
        f = csv.reader(file)
        isFirstRow = True
        for row in f:
            if isFirstRow:
                isFirstRow = False
                firstRow = row
            else:
                image_name = row[0].split('/')[1]
                # expression is at row[6]
                image_dir[image_name] = int(row[6])
    return image_dir

if __name__ == '__main__':
    image_names = os.listdir('resized_images_128')
    image_directory = buildDir('automatically_annotated.csv')
    image_folder = './resized_images_128'
    with open('data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for image_name in image_names:
            expression = image_directory[image_name]
            image_path = image_folder + '/' + image_name
            with open(image_path, 'rb') as f:
                with Image.open(f) as im:
                    pixels = list(im.getdata())
                    rpixels = [p[0] for p in pixels]
                    gpixels = [p[1] for p in pixels]
                    bpixels = [p[2] for p in pixels]
                    img = np.zeros([128, 128, 3], dtype=np.uint8)
                    img[:, :, 0] = np.array(rpixels).reshape([128,128])
                    img[:, :, 1] = np.array(gpixels).reshape([128,128])
                    img[:, :, 2] = np.array(bpixels).reshape([128,128])
                    plt.imshow(np.array(pixels))
                    row = rpixels + gpixels + bpixels + [expression]
                    writer.writerow(row)


