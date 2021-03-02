# import the necessary packages
from nms import non_max_suppression_slow, non_max_suppression_fast
import numpy as np
import cv2
import time
import xml.etree.ElementTree as ET
import pandas as pd
import csv

dfr=pd.read_csv('test_excel.csv')
dft=pd.read_csv('test_excel1.csv')
dfc=dfr.append(dft)
print(dfc)
def merge(dfr,dft):
## Coordinates extraction
    lengthr=len(dfr.index)
    lengtht=len(dft.index)
    boxes=[]
    box=[]
    label=[]
    labels=[]
    confidence=[]
    i=0
    for i in range(lengthr):
        x1=dfr.iloc[i,3]
        y1=dfr.iloc[i,4]
        x2 = dfr.iloc[i, 5]
        y2 = dfr.iloc[i, 6]
        conf=float(dfr.iloc[i,8])
        boxr=[int((x1+x2)/2),int((y1+y2)/2),int(abs(y1-y2)),int(abs(x1-x2))]
        boxes=np.append(boxes,boxr,axis=0)
        confidence=np.append(confidence,conf)
        labelr=[dfr.iloc[i, 6]]
        labels=np.append(labels,labelr,axis=0)


    for i in range(lengtht):
        x1 = dft.iloc[i, 3]
        y1 = dft.iloc[i, 4]
        x2 = dft.iloc[i, 5]
        y2 = dft.iloc[i, 6]
        conf = float(dft.iloc[i, 8])
        boxt=[int((x1+x2)/2),int((y1+y2)/2),int(abs(y1-y2)),int(abs(x1-x2))]
        boxes = np.append(boxes, boxt, axis=0)
        confidence = np.append(confidence, conf)
        labelt = [dft.iloc[i, 6]]
        labels = np.append(labels, labelt, axis=0)

    labels=np.reshape(labels,(-1,1))
    boxes=list(boxes)
    confidence=list(confidence)
    boxes=[boxes[x:x+4] for x in range(0, len(boxes), 4)]


    return boxes,confidence

boxes,confidence=merge(dfr,dft)
print(boxes)
yu= cv2.dnn.NMSBoxes(boxes, confidence, 0.1, 0.1)
print(yu)

dfc=dfc.drop(dfc.index[1])
print(dfc)

