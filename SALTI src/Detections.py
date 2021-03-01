'''
Class for storing all detections
'''
import cv2
import numpy as np


class Detections():
    def __init__(self, boundingBoxes, classes, confidences):
        self.bboxs = boundingBoxes
        self.classIds = classes
        self.confs = confidences

# def nms(boxes, confidences, classes, conf_threshold=0.5, nms_threshold=0.3 ):
#     # Non maximum suppression, will give indices to keep
#     indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
#     indices = indices.reshape((-1,))
#     return boxes[indices], confidences[indices], classes[indices]

def nms(boxes, confidences, classes, conf_threshold=0.5, nms_threshold=0.3 ):
    # Non maximum suppression, will give indices to keep
    indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
    to_keep = [i[0] for i in indices]
    return [boxes[i] for i in to_keep], [classes[i] for i in to_keep], [confidences[i] for i in to_keep]


