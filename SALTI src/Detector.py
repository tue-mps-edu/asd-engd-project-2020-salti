import cv2
import config as cfg

class Detector():
    def __init__(self, classnames, weights):
        self.classnames = classnames
        self.weights = weights
        pass


class ColorDetector(Detector):
    def __init__(self, classnames, weights):
        super().__init__(classnames,weights)
        pass

class ThermalDetector(Detector):
    def __init__(self):
        pass

class YOLOv3_320(ColorDetector):
    def __init__(self,classnames, weights):
        # Define the network
        net = cv2.dnn.readNetFromDarknet(cfg_C.dir_cfg,cfg_C.dir_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get classes
        classNames = getclasses(cfg_C.dir_classes)

        super().__init__(classnames,weights)

    def detect(self, image):
        #return Detection(boxes, classes, confidences)
        pass 

class YoloJoeHeller(ThermalDetector):
    def __init__(self):
        pass

det = ColorDetector(1,2)

print('pausehere')
