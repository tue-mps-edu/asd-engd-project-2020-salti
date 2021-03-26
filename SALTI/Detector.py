import numpy as np
from . import Detections
from . import utils_thermal # set ONNX_EXPORT in models.py
from . import utils_thermal


class Detector(object):
    def __init__(self, type=None, conf_threshold=0):
        '''
        Detector class generator

        Arguments:
            (str) algorithm to create detector for
            (float) confidence threshold for approving detections
        '''

        if type == 'RGB':               # Select an RGB method
            self.net = YOLOv3_320(conf_threshold)
        elif type == 'Thermal':         # Select a Thermal method
            self.net = YoloJoeHeller(conf_threshold)
        else:
            raise ValueError(type)


    def detect(self, img):
        '''Detect the objects in a single image'''
        return self.net.detect(img)

    def get_classes(self):
        '''Get the classes file used in the detector'''
        return self.net.get_classes()


class YOLOv3_320():
    def __init__(self, conf_threshold):
        '''
        RGB YOLO v3 object detection from:

        YOLO V3 tutorial:
        https://www.youtube.com/watch?v=GGeF_3QOHGE&list=PLMoSUbG1Q_r8nz4C5Yvd17KaXy8p0ufPH&index=1

        Links to weights:
        https://pjreddie.com/darknet/yolo/
        '''
        self.__conf_threshold = conf_threshold
        self.__dir_classes = 'config_rgb/coco-rgb.names'
        self.__dir_cfg = 'config_rgb/yolov3-rgb.cfg'
        self.__dir_weights = 'config_rgb/yolov3-rgb.weights'
        self.__whT = 320  # width & height of the image input into YOLO (standard resolution, square)
        self.__net = cv2.dnn.readNetFromDarknet(self.__dir_cfg, self.__dir_weights)
        self.__net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.__classnames = self.__load_classes()

    def get_classes(self):
        """Get classes from private variable"""
        return self.__classnames

    # Get classes
    def __load_classes(self):
        '''Load classnames from file'''
        classNames = []
        with open(self.__dir_classes, 'r') as f:  # using WITH function takes away the need to use CLOSE file function
            classNames = f.read().rstrip('\n').split(
                '\n')  # rstrip strips off the ("content here") and split splits off for("content here")
        return classNames

    def __getobjects(self,outputs, img_rgb):
        '''
        Get objets from detector outputs

        Arguments:
            outputs from detector
            (opencv img) image
        Returns:
            (list) Bounding Boxes
            (list) class identifiers
            (list) confidences
        '''
        hT, wT, cT = img_rgb.shape
        bboxs, classIds, confs = [], [], []

        for output in outputs:
            for det in output:
                scores = det[5:]
                # Select class with highest confidence level
                classId = np.argmax(scores)
                confidence = scores[classId]
                # Filter detections based on confidence threshold
                if confidence > self.__conf_threshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bboxs.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        return bboxs, classIds, confs

    def detect(self, image):
        '''
        Detect objects in RGB image
        Arguments:
            (opencv image) image to perform detection on
        Returns:
            (Detections) Output detections from algorithm
        '''
        # Send image through OpenCV neural net
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.__whT, self.__whT), [0, 0, 0], 1, False)
        self.__net.setInput(blob)
        layerNames = self.__net.getLayerNames()
        outputNames = [layerNames[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]
        outputs = self.__net.forward(outputNames)

        # Get all objects from the outputs that are above confidence level
        boxes, classes, confidences = self.__getobjects(outputs, image)
        return Detections(boxes, classes, confidences)


class YoloJoeHeller():
    def __init__(self, confThreshold):
        '''
        Thermal YOLO BBOX object detection by Joe Hoeller:
        https://github.com/joehoeller/Object-Detection-on-Thermal-Images

        Arguments:
            (flt) confidence threshold for approving bboxs
        '''
        self.__whT = 320  # width & height of the image input into YOLO (standard resolution, square)
        self.__confThreshold = confThreshold  # Confidence threshold for approval of detection
        self.__dir_cfg = 'config_thermal/yolov3-spp-r.cfg'
        self.__dir_data = 'config_thermal/coco-thermal.data'
        self.__dir_weights= 'config_thermal/yolov3-thermal-best.pt'
        self.__img_size=416
        self.__half=False
        self.__device=''
        self.__view_img=False

        with torch.no_grad():
            img_size = (
            320, 192) if ONNX_EXPORT else self.__img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
            weights, half, view_img = self.__dir_weights, self.__half, self.__view_img

            # Initialize the device you're running on
            device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.__device)

            # Initialize detection model
            model = Darknet(self.__dir_cfg, img_size)

            # Load weights from harddrive as specified in configuration file
            attempt_download(weights)
            if weights.endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(weights, map_location=device)['model'])
            else:  # darknet format
                _ = load_darknet_weights(model, weights)

            # Second-stage classifier
            classify = False
            if classify:
                modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
                modelc.to(device).eval()

            # Evaluation mode
            model.to(device).eval()

            # Export mode
            if ONNX_EXPORT:
                img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
                torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

                # Validate exported model
                import onnx
                model = onnx.load('weights/export.onnx')  # Load the ONNX model
                onnx.checker.check_model(model)  # Check that the IR is well formed
                print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
                return

            # Half precision
            half = half and device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                model.half()

            # Get classes and colors
            classes = load_classes(parse_data_cfg(self.__dir_data)['names'])
        #        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        self.__model=model
        self.__classnames=classes
        self.__device=device

    def get_classes(self):
            return self.__classnames

    def __padd_and_normalize_image(self,img0):
        '''Add padding to the image'''

        # Padded resize
        img = letterbox(img0, new_shape=self.__img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.__half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        img = torch.from_numpy(img).to(self.__device)

        return img, img0

    def __convert_tensor_to_lists(self,preds, img, im0):
        '''Convert the detections from Tensor into a list as used by SALTI'''

        bboxs, confs, classes = [], [], []
        for i, det in enumerate(preds):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Save the bounding box coordinates [Top left coordinates, width, height]
                for *xyxy, conf, _, cls in det:
                    t = np.squeeze(xyxy)  # xyxy is x_topLeft,y_topLeft,x_bottomRight,y_bottomRight
                    bboxs.append([int(t[0]), int(t[1]), int(abs(t[0] - t[2])), int(abs(t[1] - t[3]))])
                    confs.append(float(conf))  # Confidence factor
                    classes.append(int(cls))  # Class names
        return bboxs, confs, classes

    def detect(self, img):
        '''
        Run single image through the detection algorithm

        Arguments:
            (opencv img) image to label

        Outputs:
            (Detections) Detections filtered on Confidence
        '''

        with torch.no_grad():

            img, img0 = self.__padd_and_normalize_image(img)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.__model(img)[0]

            if self.__half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.__confThreshold)
            bboxs, confs, classes = self.__convert_tensor_to_lists(pred, img, img0)

        return Detections(bboxs, classes, confs)

