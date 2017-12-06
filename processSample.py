"""
    build a csv for all info about sample images we take
    samples are from folder 1
"""

import csv
import os

samplesList = os.listdir('Automatically_Annotated_Images/1')

filesInSample = dict()
firstRow = None
with open('automatically_annotated.csv') as csvFile:
    f = csv.reader(csvFile)
    isFirstRow = True
    for row in f:
        if isFirstRow:
            isFirstRow = False
            firstRow = row
            continue
        if row[0].startswith('1/'):
            fileName = row[0][2:]
            filesInSample[fileName] = row
        else:
            break

with open('sampleInfo.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(firstRow)
    for fileName,row in sorted(filesInSample.items()):
        writer.writerow(row)

