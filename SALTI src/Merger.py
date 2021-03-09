from Detections import *
import cv2

class Merger():
    def __init__(self,conf_threshold,nms_threshold):
        self.confThreshold=conf_threshold
        self.nmsThreshold=nms_threshold

    def NMS(self, detection):
        # Non maximum suppression, will give indices to keep
        indices = cv2.dnn.NMSBoxes(detection.boxes, detection.confidences, self.confThreshold, self.nmsThreshold)
        to_keep = [i[0] for i in indices]
        return Detections([detection.boxes[i] for i in to_keep],
                          [detection.classes[i] for i in to_keep],
                          [detection.confidences[i] for i in to_keep])

def test_merger():
    b=Detections([[1,2,3,4],[1,2,3,4]],[1,1],[0.1,0.7])
    a=Merger(0.5,0.5)
    c=a.NMS(b)
    print(c.boxes)
    print(c.classes)
    print(c.confidences)
    print('length'+str(len(c.classes)))
    assert(len(c.classes)==1)
