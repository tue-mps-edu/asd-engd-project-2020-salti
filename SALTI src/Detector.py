import cv2
import numpy as np
from Detections import *

# from config import *

class Detector():
    def __init__(self, classnames, weights):
        self.classnames = classnames
        self.weights = weights
        pass


class ColorDetector(Detector):
    def __init__(self, classnames, weights):
        super().__init__(classnames,weights)
        pass

# class ThermalDetector(Detector):
#     def __init__(self):
#         pass
#
class YOLOv3_320(ColorDetector):
    def __init__(self):
        self.dir_classes = 'Yolo_config/coco-rgb.names'
        self.dir_cfg = 'Yolo_config/yolov3-rgb.cfg'
        self.dir_weights = 'Yolo_config/yolov3-rgb.weights'
        self.whT = 320  # width & height of the image input into YOLO (standard resolution, square)
        self.confThreshold = 0.3     # Confidence threshold for approval of detection

        self.net = cv2.dnn.readNetFromDarknet(self.dir_cfg,self.dir_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.getclasses()
        # Define the network




    # Get classes
    def getclasses(self):
        classNames = []
        with open(self.dir_classes, 'r') as f:  # using WITH function takes away the need to use CLOSE file function
            classNames = f.read().rstrip('\n').split(
                '\n')  # rstrip strips off the ("content here") and split splits off for("content here")
        self.classNames = classNames

    def getobjects(self,outputs, img_rgb, conf_threshold):
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
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bboxs.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        return bboxs, classIds, confs

    def detect(self, image):
        # Send image through OpenCV neural net
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, False)
        self.net.setInput(blob)
        layerNames = self.net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(outputNames)

        # Get all objects from the outputs that are above confidence level
        boxes, classes, confidences = self.getobjects(outputs, image, self.confThreshold)

        return Detections(boxes, classes, confidences)

        # Do non-maximum suppression for remaining boxes
        # boxes, classes, confidences = nms(boxes, confidences, classes, cfg_C.confThreshold, cfg_C.nmsThreshold)

        # Store in container
        # det = Detections(boxes, classes, confidences)

        # return boxes, confidences, classes

'''
    def detect(self, image):
        #return Detection(boxes, classes, confidences)
        pass 
#
# class YoloJoeHeller(ThermalDetector):
#     def __init__(self):
#         pass

det = ColorDetector(1,2)
'''

a=cv2.imread(r'C:\Users\20204916\Documents\GitHub\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\visible\I00041.jpg')
b=YOLOv3_320()
c=b.detect(a)
print(c)