import cv2
import os
import numpy as np
import datetime
import time
import math


def get_iou(pred_box, gt_box):
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def MultiScale(images, scaleValue):
    if scaleValue <= 1 or scaleValue >= 2:
        return
    listOfResult = np.empty((4,0))
    Scale = scaleValue - 1
    timesLoop = 1 / Scale
    images = cv2.resize(images, (images.shape[1] * (scaleValue - 1), images.shape[0] * (scaleValue - 1)))
    # listOfResult = np.add(listOfResult, np.array([x, y, w, h]), axis = 0)

    return listOfResult


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def detectMultiScale_custom(gray_images, scaleFactor, Step):
    if scaleFactor <= 2 and scaleFactor > 1:
        Scale = scaleFactor - 1
        listOfResult = np.array([[0,0,0,0]], dtype=np.float)
        scale = 0
        while (scale < 100):
            scale += Scale
            scale1 = 1 - scale / 100
            img = gray_images.copy()
            if scale1 * img.shape[1] < 30:
                break
            if scale1 * img.shape[1] > 70:
                continue
            img = cv2.resize(img, (int(scale1 * img.shape[1]), int(scale1 * img.shape[0])))
            listOfResult = sildeWindown(img, listOfResult, Step, scale)
        return listOfResult
    else:
        return

def sildeWindown(img, listOfResult, Step, Scale):
    Cus_Cascade = cv2.CascadeClassifier('data\haarcascade\myfacedetector.xml')
    w, h = Cus_Cascade.getOriginalWindowSize()
    w, h = w + 1, h + 1
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1], Step):
            if i + w <= img.shape[0] and j + h <= img.shape[1]:
                crop_img = img[(i - 1): (i + w), (j - 1): (j + h)]
                ObjectDetect = Cus_Cascade.detectMultiScale(crop_img, 10000, minNeighbors= 1
                                    , maxSize=(w, h), minSize=(w, h))
                try:
                    if ObjectDetect != ():
                        for x, y, w1, h1 in ObjectDetect:
                            #cv2.rectangle(crop_img, (x,y), (x + w1, y + h1), (244,42,2), 2)
                            #cv2.imshow('crop_img', cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR))
                            scale1 = 1 + (Scale / (100 - Scale))
                            listOfResult = np.append(listOfResult, np.array(
                                [[(j + x + 1) * scale1, (i + y + 1) * scale1, (w1) * scale1, (h1) * scale1]]), axis=0)
                            #print(Scale, scale1, int(i), int(y), int(w1), int(h1))
                            #cv2.rectangle(img, (int(j + x + 1), int(i + y + 1)),
                            #              (int(j + w1 + x + 1), int(i + h1 + y + 1)), (255,0,255), 4)
                            #cv2.imshow('img_Resize', img)
                            #cv2.imshow('crop_img', crop_img)
                            #cv2.waitKey()
                except:
                    continue
    return listOfResult


path = r'C:\Users\conso\Downloads\HaarCascade-Training-Facas\Haar Training\training\positive\rawdata'
path_detect = 'Detector'
listOfResult = []
with open(r"C:\Users\conso\Downloads\HaarCascade-Training-Facas\Haar Training\training\positive\info.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        listOfResult.append(line.split()[1:])
sum = 0
for i in listOfResult:
    sum += int(i[0])
count = 0
TP, FP = 0, 0


for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG") or filename.endswith("bmp"):
        img = cv2.imread(os.path.join(path, filename))
        Scale = 500 / img.shape[1]
        img1 = cv2.resize(img, (int(Scale * img.shape[1]),int(Scale * img.shape[0])))
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # now = str(datetime.datetime.now().time())
        # timeNow = now.split(':')
        objectDetected = detectMultiScale_custom(gray, 1.01, 2)
        objectDetected = objectDetected / Scale
        if len(objectDetected) > 2:
            for i in range(1, len(objectDetected)):
                cv2.rectangle(img, (int(objectDetected[i][0]), int(objectDetected[i][1])), (int(objectDetected[i][2])
                                    + int(objectDetected[i][0]), int(objectDetected[i][3]) + int(objectDetected[i][1]))
                              ,(255, 0, 0), 6)
        objectDetectedNMS = non_max_suppression_fast(objectDetected, 0.5)
        for (x, y, w, h) in objectDetectedNMS:
            cv2.rectangle(img, (int(x), int(y)), (int(w) + int(x), int(h) + int(y)), (255, 255, 0), 2)
        objectDetectedNMS = np.delete(objectDetectedNMS, objectDetectedNMS.shape[0] - 1, axis=0)
        print(objectDetectedNMS, len(objectDetectedNMS))
        print("this", listOfResult[count])
        if len(objectDetectedNMS) > 0:
            x1 = objectDetectedNMS[:, 0]
            y1 = objectDetectedNMS[:, 1]
            x2 = objectDetectedNMS[:, 2]
            y2 = objectDetectedNMS[:, 3]
            for i in range(len(x1)):
                pred_box = np.array([x1[i], y1[i], x2[i] + x1[i], y2[i] + y1[i]])
                isNot = 0
                for itemx in range(int(listOfResult[count][0])):
                    cv2.rectangle(img, (int(listOfResult[count][1 + itemx * 4]), int(listOfResult[count][2 + itemx * 4])
                                        ), (int(listOfResult[count][3 + itemx * 4]) + int(listOfResult[count][1 + itemx * 4]),
                                        int(listOfResult[count][4 + itemx * 4]) + int(listOfResult[count][2 + itemx * 4]))
                                        ,(0,0,255), 4)
                    gt_box = np.array(
                        [int(listOfResult[count][1 + itemx * 4]), int(listOfResult[count][2 + itemx * 4])
                            , int(listOfResult[count][3 + itemx * 4]) + int(listOfResult[count][1 + itemx * 4]),
                         int(listOfResult[count][4 + itemx * 4]) + int(listOfResult[count][2 + itemx * 4])])

                    # print(pred_box, gt_box, get_iou(gt_box, pred_box))
                    if get_iou(gt_box, pred_box) >= 0.5:
                        TP += 1
                        listOfResult[count].append(-1)
                        isNot = 1
                        break
                if isNot == 0:
                    FP += 1

        """
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        """
        print(TP, FP, count)
        count += 1
        del x, y, w, h

cv2.destroyAllWindows()
sum = 0
for i in listOfResult:
    sum += int(i[0])
print("ReCall: " + str((TP/sum) * 100) + "% with TP: " + str(TP) + " and totalOfFace: " + str(sum))
try:
    print("Predection: " + str((TP / (TP + FP)) * 100) + "% with TP: " + str(TP) + " and FP: " + str(FP))
except:
    print(TP, FP)