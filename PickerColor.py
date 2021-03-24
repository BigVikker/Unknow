import cv2
import numpy as np

img = cv2.imread('Img/images (5).jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
light = (255, 255, 255)
dark_light = (0, 0, 125)
mask = cv2.inRange(hsv, dark_light, light)
#result = cv2.inRange(img, img, mask = mask)
cv2.imshow('result', mask)
cv2.waitKey()
cv2.destroyAllWindows()