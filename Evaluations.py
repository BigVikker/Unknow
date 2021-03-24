import cv2
import os
import sys
import numpy as np
import math
import copy


#import torch
#from torchvision.ops import nms

positive = 0
TF = 0
sumPlate = 0
location1, location2 = [], []
flag = 0

"""
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
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
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
            # delete all indexes from the index list that are in the
            # suppression list
        idxs = np.delete(idxs, suppress)
        # return only the bounding boxes that were picked
        return boxes[pick]
"""
def nms1(dets, thresh):
    print(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	"""if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	"""
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
	# list
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
        # overlap each
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


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

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global location1, location2, flag
        if flag % 2 == 0:
            location1.append([x, y, 0])
            #cv2.circle(img, (x, y), 2, (0,255,0), 2)
        elif flag % 2 == 1:
            location2.append([x, y, 0])
            #cv2.circle(img, (x, y), 2, (0 ,0, 255), 2)
            if location1 != None and location2 != None:
                pass
                cv2.rectangle(img, (location1[0][0], location1[0][1]), (location2[0][0], location2[0][1]), (0, 0, 255), 3)
                print(location1, location2 )
                #cv2.destroyWindow('img')
        flag += 1

listOfResult = []
with open("Result/ResultPerAndRecall.txt") as fp:
    Lines = fp.readlines()
    for line in Lines:
        listOfResult.append(line.strip())
listOfPosNumberPlate = []
for i in listOfResult:
    stringPos = i.replace(']', '')
    stringPos = stringPos.replace('[', '')
    listPos = stringPos.split(',')
    listPos[2] = listPos[2][3:]
    listPos[4] = '0'
    listOfPosNumberPlate.append(listPos)
FN = 0
directory = 'Img'
listFileName = []
count = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):
        listFileName.append(filename)

listFileName.sort()
for item in range(len(listFileName)):
    img = cv2.imread(os.path.join(directory, listFileName[item]))
    confidence_score = []
    cv2.useOptimized()
    for i in range(5,15,5):
        img1 = img.copy()
        scale = 1 + (0.1 * i)
        img1 = cv2.resize(img1, (int(img1.shape[1] * scale) , int(scale * img1.shape[0])))
        img1 = cv2.GaussianBlur(img1, (3, 3), 0)
        img1 = cv2.addWeighted(img1, 1.3, np.zeros(img1.shape, img1.dtype), -50, 2)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        listOfBox = np.empty((0, 4))
        for valuesThresh in range(5, 125, 10):
            thresh = cv2.Canny(gray, valuesThresh, 499)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                for scale in range(1, 8):
                    flagsDected = False
                    tiLe = 0.01
                    tiLe += tiLe * scale
                    approx = cv2.approxPolyDP(cnt, tiLe * cv2.arcLength(cnt, True), True)
                    if len(approx) <= 4 and len(approx) <= 10:
                        area = cv2.contourArea(cnt)
                        x, y, w, h = cv2.boundingRect(cnt)
                        x = x - int((i / (i + 10) * x))
                        y = y - int((i / (i + 10) * y))
                        w = w - int((i / (i + 10) * w))
                        h = h - int((i / (i + 10) * h))
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        reChange = reorder(box)
                        height = int(
                            length2PtsPixels(reChange[3][0][0], reChange[3][0][1], reChange[1][0][0],
                                             reChange[1][0][1]))
                        width = int(
                            length2PtsPixels(reChange[1][0][0], reChange[1][0][1], reChange[2][0][0],
                                             reChange[2][0][1]))
                        if area >= img1.shape[1] * img1.shape[0] * 0.01 and area <= img1.shape[
                            1] * img1.shape[0] * 0.9 and width >= height * 2 and width <= height * 6:
                            positive += 1

                            gt_box = np.array(
                                [int(listOfPosNumberPlate[item][0]), int(listOfPosNumberPlate[item][1])
                                    , int(listOfPosNumberPlate[item][2]),
                                 int(listOfPosNumberPlate[item][3])])
                            pred_box = np.array([x, y, x + w, y + h])
                            if get_iou(pred_box, gt_box) > 0.2:
                                TF += 1
                                listOfPosNumberPlate[item][4] = str(1)
                                confidence_score.append(get_iou(pred_box, gt_box))
                                boxObj = np.array([[x, y, (x + w), (y + h)]])
                                listOfBox = np.append(listOfBox, boxObj, axis=0)
                                print(boxObj)
                            # .append([x, y, (x + w), (y + h)])

    #cv2.waitKey()
    """nms_box = nms1(listPos, 0.5)
    for i in nms_box:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,255), 2)"""
    try:
        cv2.rectangle(img, (int(boxDected[0][0][0]), int(boxDected[0][0][1])),
                      (int(boxDected[0][0][2]), int(boxDected[0][0][3])), (0, 255, 255), 2)

        cv2.putText(img, '{0:.4f}'.format(boxDected[1][0]), (int(boxDected[0][0][0]), int(boxDected[0][0][1])),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 222), 2)
    except:
        pass
    """for i in boxDected[0]:
        for j in i:
           print(j)"""


    """for box in ROI[0]:
        # img_crop = img[box[0]: box[2], box[1]: box[3]]
        # result = cv2.inRange(img, img, mask = mask)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        """

    """for box in non_max_suppression_fast(listOfBox, 0.5):
        positive += 1
        gt_box = np.array([int(listOfPosNumberPlate[item][0]), int(listOfPosNumberPlate[item][1])
                              , int(listOfPosNumberPlate[item][2]), int(listOfPosNumberPlate[item][3])])
        pred_box = box
        if get_iou(pred_box, gt_box) > 0.25:
            TF += 1
            listOfPosNumberPlate[item][4] = str(1)
            confidence_score.append(get_iou(pred_box, gt_box))"""
    cv2.destroyAllWindows()


#cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
#cv2.rectangle(img, (location1[0], location1[1]), (location2[0], location2[1]), (0, 0, 255), 3)
print(TF, positive)
try:
    print("perdection: " + str((TF / positive) * 100) + " %")
except:
    print("perdection : 0%")
TF = 0
for i in listOfPosNumberPlate:
    if i[4] != '0':
        TF += 1

print(TF, len(listOfPosNumberPlate))
try:
    print("Recall: " + str((TF / len(listOfPosNumberPlate)) * 100) + " %")
except:
    print("devine by zero")

cv2.destroyAllWindows()


