import os
import cv2
import random
import numpy as np
from shutil import copyfile


position = []

def numberSave(stringNumber):
    pass





listOfResult = []
with open("Result/ResultPerAndRecall.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        listOfResult.append(line.strip())

directory = 'Img'
listFileName = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):
        listFileName.append(filename)
listFileName.sort()
for item in range(len(listFileName)):
    img = cv2.imread(os.path.join(directory, listFileName[item]))

    stringPos = listOfResult[item].replace(']','')
    stringPos = stringPos.replace('[', '')
    listPos = stringPos.split(',')
    cv2.rectangle(img, (int(listPos[0]), int(listPos[1])),
                  (int(listPos[2][3:]), int(listPos[3])), (0,255,255), 2)
    print(str(item+1) + str(listFileName[item]) +" bien So xe"+ str(listPos))
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
