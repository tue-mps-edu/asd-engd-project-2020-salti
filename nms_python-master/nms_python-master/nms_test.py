# import the necessary packages
from nms import non_max_suppression_slow, non_max_suppression_fast
import numpy as np
import cv2
import time
import xml.etree.ElementTree as ET
import pandas as pd

dfr=pd.read_csv('test_excel.csv')
dft=pd.read_csv('test_excel1.csv')
def merge(dfr,dft):
## Coordinates extraction
    lengthr=len(dfr.index)
    lengtht=len(dft.index)
    i=1
    boxes=[]
    box=[]
    label=[]
    labels=[]
    for i in range(lengthr):
        x1=dfr.iloc[i,2]
        y1=dfr.iloc[i,3]
        x2 = dfr.iloc[i, 4]
        y2 = dfr.iloc[i, 5]
        boxr=[x1,y1,x2,y2]
        boxes=np.append(boxes,boxr,axis=0)

        labelr=[dfr.iloc[i, 6]]
        labels=np.append(labels,labelr,axis=0)


    for i in range(lengtht):
        x1 = dft.iloc[i, 2]
        y1 = dft.iloc[i, 3]
        x2 = dft.iloc[i, 4]
        y2 = dft.iloc[i, 5]
        boxt = [x1, y1, x2, y2]
        boxes = np.append(boxes, boxt, axis=0)

        labelt = [dft.iloc[i, 6]]
        labels = np.append(labels, labelt, axis=0)

    boxes= np.reshape(boxes,(-1,4))
    labels=np.reshape(labels,(-1,1))
    boxes=boxes.astype(int)

    images = [("images/Parking.jpg", boxes,labels)]
    iter_num= 1
    images = images*iter_num  # change the iterations to compare the two nms method

    t1 = time.time()

    # loop over the images
    for (i, (imagePath, boundingBoxes,labels)) in enumerate(images):
        # load the image and clone it
        # print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
        image = cv2.imread(imagePath)
        orig = image.copy()

        # loop over the bounding boxes for each image and draw them
        for (startX, startY, endX, endY) in boundingBoxes:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # perform non-maximum suppression on the bounding boxes
        # pick = non_max_suppression_slow(boundingBoxes, 0.3)

        pick,pick_labels = non_max_suppression_fast(boundingBoxes,labels, probs=None, overlapThresh=0.3)
        # print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

        # loop over the picked bounding boxes and draw them
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # display the images
        # cv2.imshow("Original" + i, orig)
        # cv2.imshow("After NMS" + i, image)
        # cv2.waitKey(0)

        # save the images
        cv2.imwrite("images/Original_test" + str(i) + ".jpg", orig)
        cv2.imwrite("images/After_NMS_test" + str(i) + ".jpg", image)


    return pick,pick_labels

box,label=merge(dfr,dft)
print(box,label)
# t2 = time.time()
# print('cost {} ms to process {} images'.format((t2 - t1)*1000, len(images)))
# boxes,labels = extract("test_excel.csv")
# boxes1,labels1= extract("test_excel1.csv")
# a=boxes
# b=boxes1
# bbox=np.append(a,b,axis=0)
# bbox=bbox.astype(int)
# print(type(bbox))

# construct a list containing the images that will be examined
# along with their respective bounding boxes
