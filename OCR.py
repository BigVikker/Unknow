import cv2
import pytesseract
import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt


def length2PtsPixels(x, y, x1, y1):
    return math.sqrt((math.pow(x - x1, 2)) + (math.pow(y - y1, 2)))

def plateCharacter(strNumberplate):
    s1 = ''
    for i in strNumberplate:
        if i.isalpha():
            if i.isupper():
                s1 += i
        if i.isdigit():
            s1 += i
    return s1

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
numberCorrectOCR = 0
OCR80 = 0
OCR80FromCas = 0
numberCorrectOCR_fromCas = 0
def check(data, template, flags):
    data = plateCharacter(data)
    template = plateCharacter(template)
    if data == template:
        global numberCorrectOCR
        numberCorrectOCR += 1
        if flags == True:
            global numberCorrectOCR_fromCas
            numberCorrectOCR_fromCas += 1

    count = 0
    for i in template:
        if data.find(i) == -1:
            count += 1
            continue
    if (count / len(template)) >= 0.75:
        global OCR80
        OCR80 += 1
        if flags == True:
            global OCR80FromCas
            OCR80FromCas += 1

numberOfCasDect = 0
sumImgCanRead = 0
sumImgHaveROI = 0

listOfResult = []
with open("Result/numberPlateResult.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        listOfResult.append(line.strip())

directory = 'Img'
listFileName = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".Jpeg"):
        listFileName.append(filename)
listFileName.sort()
for i in range(0, len(listFileName)):
    cv2.useOptimized()
    sumImgCanRead += 1
    img = cv2.imread(os.path.join(directory, listFileName[i]))
    if img.shape[1] < 300:
        scale = 600 / img.shape[1]
        img = cv2.resize(img, (int(scale * img.shape[1]), int(scale * img.shape[0])))
    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), -0.5, 0)

    numberPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    numberPlate = numberPlateCascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=20)
    if len(numberPlate) != 0:
        sumImgHaveROI += 1
        numberOfCasDect += 1
    for (x, y, w, h) in numberPlate:
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        img_crop = img[y:y + h, x:x + w]
        scale = 400 / img_crop.shape[1]
        imgWarpColored = cv2.resize(img_crop,
                                    (int(img_crop.shape[1] * scale), int(img_crop.shape[0] * scale)))
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        gray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        invert = 255 - thresh
        #cv2.imshow('Bien So', invert)
        config = r'--oem 1 --psm 7 outputbase'
        data = pytesseract.image_to_string(invert, lang='eng', config=config)
        check(data, listOfResult[i], True)
    if len(numberPlate) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.Canny(gray, 20, 200)
        contours, h = cv2.findContours(thresh, 1, 2)
        largest_rectangle = [0, 0]
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if area > largest_rectangle[0] and area >= img.shape[1] * img.shape[0] * 0.01 and area <= img.shape[1] * \
                        img.shape[0] * 0.9:
                    largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
        if largest_rectangle != [0, 0]:
            sumImgHaveROI += 1
            # minumBox bounding
            rect = cv2.minAreaRect(largest_rectangle[1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # sort by X
            reChange = reorder(box)
            # Warp Prespective
            height = int(length2PtsPixels(reChange[3][0][0], reChange[3][0][1], reChange[1][0][0], reChange[1][0][1]))
            width = int(length2PtsPixels(reChange[1][0][0], reChange[1][0][1], reChange[2][0][0], reChange[2][0][1]))
            pts1 = np.float32(reChange)
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

            # increase Contrast
            #imgWarpColored = cv2.addWeighted(imgWarpColored, 1.5, np.zeros(imgWarpColored.shape, imgWarpColored.dtype),
                                             #-0.5, 0)
            # resize
            scale = 200 / imgWarpColored.shape[1]
            # scale = 1
            imgWarpColored = cv2.resize(imgWarpColored,
                                        (int(imgWarpColored.shape[1] * scale), int(imgWarpColored.shape[0] * scale)))

            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            # cv2.imshow('DANH DAU DOI TUONG', img)

            # -----------------------------------------------------

            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            #imgWarpColored = cv2.addWeighted(imgWarpColored, 1.2, np.zeros(img.shape, img.dtype), -0.9, 0)
            gray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            invert = 255 - thresh
            #cv2.imshow('palete number ', thresh)
            config = r'--oem 1 --psm 7 outputbase'
            data = pytesseract.image_to_string(invert, lang='eng', config=config)
            check(data, listOfResult[i], flags= False)



    #cv2.imshow('DANH DAU DOI TUONG', img)
    #cv2.waitKey()
    cv2.destroyAllWindows()


Cas = (sumImgCanRead, numberOfCasDect, OCR80FromCas, numberCorrectOCR_fromCas)
noCas = (0, (sumImgHaveROI - numberOfCasDect), (OCR80 - OCR80FromCas), (numberCorrectOCR - numberCorrectOCR_fromCas))
CasStd = (2, 3, 4, 1)
noCasStd = (3, 5, 2, 3)
ind = np.arange(4)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, Cas, width, yerr=CasStd)
p2 = plt.bar(ind, noCas, width,
             bottom=Cas, yerr=noCasStd)
x = [0, 1, 2, 3]
y = [sumImgCanRead, sumImgHaveROI, OCR80, numberCorrectOCR]
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.ylabel('Images')
plt.xlabel('Correct Images')
plt.xticks(ind, ('Tổ số ảnh', 'ROI', '75%', '100%'))
plt.yticks(np.arange(0, sumImgCanRead + 20, 5))
plt.legend((p1[0], p2[0]), ('số ảnh Cas', 'số ảnh no Cas'))

plt.show()
# function to show the plot

cv2.waitKey()
cv2.destroyAllWindows()