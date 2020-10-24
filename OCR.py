import cv2
import pytesseract
import numpy as np
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


img = cv2.imread(r"C:\Users\Admin\Desktop\xulyanh\BienSoXe\vd2.jpg")
cv2.useOptimized()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('HINH ANH GOC', img)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours,h= cv2.findContours(thresh, 1, 2)
largest_rectangle= [0, 0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]




#minumBox bounding
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
imgWarpColored = cv2.addWeighted(imgWarpColored, 1.5, np.zeros(imgWarpColored.shape, imgWarpColored.dtype), -0.5, 0)
# resize
scale = 300 / imgWarpColored.shape[1]
imgWarpColored = cv2.resize(imgWarpColored, (int(imgWarpColored.shape[1] * scale), int(imgWarpColored.shape[0] * scale)))

cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
cv2.imshow('DANH DAU DOI TUONG', img)

#-----------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
gray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening
cv2.imshow('Bien So', invert)
data = pytesseract.image_to_string(invert, lang='eng', config='--oem 1 --psm 6')
print('THONG TIN NHAN DIEN: ')
print(data)
cv2.waitKey()
cv2.destroyAllWindows()

