"""
    a demo showing how to convert one image back to 128*128*3 ndarray
"""

import csv
import numpy as np

def reconstruct_img(fileName, imageNbr, height=128, width=128):
    """
    read corresponding file and generate a numpy array
    :param fileName: a csv file containing extracted data info
    :type fileName: str
    :param imageNbr: number of images to read
    :type imageNbr: int
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :return: training data of shape (height, width, 3, imageNbr) and labels of shape (imageNbr)
    :rtype: ndarray, ndarray
    """
    with open(fileName, 'r') as csvfile:
        f = csv.reader(csvfile)
        count = 0
        train_x = np.zeros([height, width, 3, imageNbr])
        train_y = np.zeros(imageNbr)
        for row in f:
            pixels = list(map(int, row))
            expression = pixels[-1]
            im = np.zeros([height, width, 3], dtype=np.uint8)
            im[:,:,0] = np.array(pixels[:height*width]).reshape([128,128])
            im[:,:,1] = np.array(pixels[height*width:height*width*2]).reshape([128, 128])
            im[:,:,2] = np.array(pixels[height*width*2:height*width*3]).reshape([128, 128])
            train_x[:,:,:,count] = im
            train_y[count] = expression
            count += 1
            if count > imageNbr:
                break
    return train_x, train_y

if __name__ == '__main__':
    train_x, train_y = reconstruct_img('sampledata.csv', 2000)