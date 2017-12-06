import csv
import numpy as np

with open('data.csv', 'r') as csvfile:
    f = csv.reader(csvfile)
    for row in f:
        pixels = list(map(int, row))
        expression = pixels[-1]
        width, height = 128, 128
        im = np.zeros([height, width, 3], dtype=np.uint8)
        im[:,:,0] = np.array(pixels[:128*128]).reshape([128,128])
        im[:,:,1] = np.array(pixels[128*128:128*128*2]).reshape([128, 128])
        im[:,:,2] = np.array(pixels[128*128*2:128*128*3]).reshape([128, 128])