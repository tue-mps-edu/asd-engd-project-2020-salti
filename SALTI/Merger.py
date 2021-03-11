from Detections import Detections
import cv2

class Merger():
    def __init__(self,conf_threshold,nms_threshold):
        self.confThreshold=conf_threshold
        self.nmsThreshold=nms_threshold

    def merge(self, detections):
        return self.NMS(detections)

    def NMS(self, detections):
        # Non maximum suppression, will give indices to keep
        indices = cv2.dnn.NMSBoxes(detections.boxes, detections.confidences, self.confThreshold, self.nmsThreshold)
        to_keep = [i[0] for i in indices]
        return Detections([detections.boxes[i] for i in to_keep],
                          [detections.classes[i] for i in to_keep],
                          [detections.confidences[i] for i in to_keep])
