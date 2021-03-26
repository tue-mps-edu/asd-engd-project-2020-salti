#from Detections import Detections
from . import Detections
import cv2

class Merger():
    def __init__(self,conf_threshold,nms_threshold):
        ''' Merger (class) that facilitates the filtering of detections
        argument 1: (float) confidence threshold
        argument 2: (float) NMS threshold
        '''
        self.confThreshold=conf_threshold
        self.nmsThreshold=nms_threshold

    def confidence_filter(self,detections):
        '''Filtering detections based on a confidence threshold
        argument 1: (Detections) input detections
        return 1: (Detections) filtered detections
        '''
        to_keep = [i for i,conf in enumerate(detections.confidences) if conf>self.confThreshold]
        return Detections([detections.boxes[i] for i in to_keep],
                          [detections.classes[i] for i in to_keep],
                          [detections.confidences[i] for i in to_keep])

    def NMS(self, detections):
        '''
        Non-maximum suppression of bounding boxes
        argument 1: (Detections) input detections
        return 1: (Detections) filtered detections
        '''
        #Non maximum suppression, will give indices to keep
        indices = cv2.dnn.NMSBoxes(detections.boxes, detections.confidences, self.confThreshold, self.nmsThreshold)
        to_keep = [i[0] for i in indices]
        return Detections([detections.boxes[i] for i in to_keep],
                          [detections.classes[i] for i in to_keep],
                          [detections.confidences[i] for i in to_keep])
