''' RGB OBJECT DETECTION
INSERT SOURCE HERE
'''

import cv2
from config import *
from detections import *
from functions import *
import os

def initialize():
    # Define the network
    net = cv2.dnn.readNetFromDarknet(cfg_C.dir_cfg,cfg_C.dir_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get classes
    classNames = getclasses(cfg_C.dir_classes)
    return net, classNames

def detect(net,classnames,image):
    # Send image through OpenCV neural net
    blob = cv2.dnn.blobFromImage(image, 1/255,(cfg_C.whT, cfg_C.whT), [0,0,0],1,False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # Get all objects from the outputs that are above confidence level
    boxes, classes, confidences = getobjects(outputs, image, cfg_C.confThreshold)
    # Do non-maximum suppression for remaining boxes
    boxes, classes, confidences = nms(boxes, confidences, classes, cfg_C.confThreshold, cfg_C.nmsThreshold)

    # Add Bounding Boxes to image
    draw_bboxs(image, boxes, confidences, classes, classnames)

    # Store in container
    det = Detections(boxes, classes, confidences)

    return det, image

def getclasses(classes_file_dir):
    classNames = []
    with open(classes_file_dir, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
        classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")
    return classNames

def getobjects(outputs, img_rgb, conf_threshold):
    hT, wT, cT = img_rgb.shape
    bboxs = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bboxs.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    return bboxs, classIds, confs
