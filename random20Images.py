import cv2
import numpy as np
import os
import math


def length2PtsPixels(x, y, x1, y1):
    return math.sqrt((math.pow(x - x1, 2)) + (math.pow(y - y1, 2)))


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


listFileName = []
directory = 'Img'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".Jpeg"):
        listFileName.append(filename)

flagsDected = False
numberOfFlagDect = 0
numberOfFlagUnDect = 0


for i in range(len(listFileName)):
    img = cv2.imread(os.path.join(directory, listFileName[i]))
    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), -0.9, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny(gray, 5, 255)
    contours, h = cv2.findContours(thresh, 1, 2)
    for scale in range(1, 8):
        flagsDected = False
        tiLe = 0.01
        tiLe += tiLe * scale
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, tiLe * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if area >= img.shape[1] * img.shape[0] * 0.01 and area <= img.shape[1] * img.shape[0] * 0.9:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    reChange = reorder(box)
                    # Warp Prespective
                    height = int(
                        length2PtsPixels(reChange[3][0][0], reChange[3][0][1], reChange[1][0][0], reChange[1][0][1]))
                    width = int(
                        length2PtsPixels(reChange[1][0][0], reChange[1][0][1], reChange[2][0][0], reChange[2][0][1]))
                    if width >= 2.5 * height and width <= 6 * height:
                        #cv2.drawContours(img, [box], 0, (255, 255, 0), 5)
                        flagsDected = True

    if flagsDected == True and numberOfFlagDect <= 20:
        cv2.imwrite("ImgUnDect/" + listFileName[i], img)
        numberOfFlagDect += 1
    if flagsDected == False and numberOfFlagUnDect <= 20:
        cv2.imwrite("ImgDect/" + listFileName[i], img)
        numberOfFlagUnDect += 1
    print(numberOfFlagDect, numberOfFlagUnDect, i)
