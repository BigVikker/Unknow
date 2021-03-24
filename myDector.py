import cv2
import numpy as np
import os



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
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def myCustomeDetoct(images, ScaleValues):
    if ScaleValues > 1 and ScaleValues <= 2:
        Scale = 1 - ScaleValues
        count = 0
        while 1:
            if Scale * count >= 1:
                break
            images1 = images.Copy()
            images1 = cv2.resize(images1, (images1.shape[1] - images1.shape[1] * Scale * count, images1.shape[0] - images1.shape[0] * Scale * count))

            count += 1
    else:
        None



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


numberPlateCascade = cv2.cv2.CascadeClassifier('data/haarcascade/myfacedetector.xml')
print(numberPlateCascade.__dir__())

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
        img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), -0.5, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        listOfRec = np.empty((0, 4), int)
        numberPlate = numberPlateCascade.detectMultiScale(gray, scaleFactor=(1.05), minNeighbors=20)
        # print(numberPlateCascade.myDector1(gray, scaleFactor=(1 + (0.001 * (j + 1))), minNeighbors=20))
        for (x, y, w, h) in numberPlate:
            listOfRec = np.append(listOfRec, np.array([[x, y, x + w, h + y]]), axis=0)
        listOfRec = non_max_suppression_fast(listOfRec, 0.5)
        if len(listOfRec) != 0:
            x1 = listOfRec[:, 0]
            y1 = listOfRec[:, 1]
            x2 = listOfRec[:, 2]
            y2 = listOfRec[:, 3]
            for i in range(len(x1)):
                cv2.rectangle(img, (x1[i], y1[i]), (x2[i], y2[i]), (0, 0, 255), 2)
                pred_box = np.array([x1[i], y1[i], x2[i], y2[i]])
                isNot = 0
                for itemx in range(int(listOfResult[count][0])):
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
        else:
            cv2.imshow(filename,img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        """
        """
        for itemx in range(int(listOfResult[count][0])):
            if int(listOfResult[count][len(listOfResult[count])- 1 - itemx]) != -1:
                cv2.imshow(filename, img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                break
        """
        count += 1
sum = 0
for i in listOfResult:
    sum += int(i[0])
print("ReCall: " + str((TP/sum) * 100) + "% with TP: " + str(TP) + " and totalOfFace: " + str(sum))
try:
    print("Predection: " + str((TP / (TP + FP)) * 100) + "% with TP: " + str(TP) + " and FP: " + str(FP))
except:
    print(TP, FP)