from Detections import *

import argparse
from utils_thermal.models_thermal import *  # set ONNX_EXPORT in models.py
from utils_thermal.datasets import *


class Detector():
    def __init__(self, classnames, weights):
        self.classnames = classnames
        self.weights = weights
        pass


# class ColorDetector(Detector):
#     def __init__(self, classnames, weights):
#         super().__init__(classnames,weights)
#         pass
#
# class ThermalDetector(Detector):
#     def __init__(self, classnames, weights):
#         super().__init__(classnames,weights)
#         pass

class YOLOv3_320(Detector):
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


#Thermal Detector Class definitiom
class YoloJoeHeller(Detector):
    def __init__(self):

        self.whT = 320  # width & height of the image input into YOLO (standard resolution, square)
        self.confThreshold = 0.3  # Confidence threshold for approval of detection
        self.nmsThreshold = 0.5  # Non-maximum suppresion threshold (lower = less number)

        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='Yolo_config/yolov3-spp.cfg', help='cfg file path')
        parser.add_argument('--data', type=str, default='Yolo_config/coco-thermal.data', help='coco.data file path')
        parser.add_argument('--weights', type=str, default='Yolo_config\yolov3-thermal.weights',
                            help='path to weights file')
        parser.add_argument('--source', type=str,
                            default=r'C:\Github\asd-pdeng-project-2020-developer\SALTI src\Data\mytestdata\RGB',
                            help='source')  # input file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=self.confThreshold, help='object confidence threshold')
        parser.add_argument('--nms-thres', type=float, default=self.nmsThreshold,
                            help='iou threshold for non-maximum suppression')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
        parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        self.opt = parser.parse_args()
        self.DEBUG_MODE = False

        if self.DEBUG_MODE:
            print(self.opt)

        with torch.no_grad():
            img_size = (
            320, 192) if ONNX_EXPORT else self.opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
            print("detect_thermal.py, line 32 update config")
            out, source, weights, half, view_img = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img
            webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

            # Initialize
            device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.opt.device)
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

            # Initialize model
            model = Darknet(self.opt.cfg, img_size)

            # Load weights
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

            # Fuse Conv2d + BatchNorm2d layers
            # model.fuse()

            # Eval mode
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
            classes = load_classes(parse_data_cfg(self.opt.data)['names'])
        #        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        self.model=model
        self.classes=classes
        self.device=device
        # return model, classes, opt, device

    def convertImage(self,img0, device, opt):
        # Padded resize
        img = letterbox(img0, new_shape=opt.img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if opt.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        img = torch.from_numpy(img).to(device)

        if self.DEBUG_MODE:
            print('original' + str(img0.shape))
            print('resized' + str(img.shape))

        return img, img0

    def getlists(self,preds, img, im0):
        bboxs, confs, classes = [], [], []
        for i, det in enumerate(preds):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, _, cls in det:  # Save the bounding box coordinates [Top left coordinates, width, height]
                    t = np.squeeze(xyxy)  # xyxy is x_topLeft,y_topLeft,x_bottomRight,y_bottomRight
                    bboxs.append([int(t[0]), int(t[1]), int(abs(t[0] - t[2])), int(abs(t[1] - t[3]))])
                    confs.append(float(conf))  # Confidence factor
                    classes.append(int(cls))  # Class names
        return bboxs, confs, classes

    def detect(self, img):
        # Run inference
        with torch.no_grad():
            # Get detections
            img, img0 = self.convertImage(img, self.device, self.opt)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]

            if self.opt.half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres)
            bboxs, confs, classes = self.getlists(pred, img, img0)

        return Detections(bboxs, classes, confs)



# a=cv2.imread(r'C:\Users\20204916\Documents\GitHub\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\visible\I00041.jpg')
# b=YOLOv3_320()
# c=b.detect(a)
# cc=Detector()
# print(c)


aa=cv2.imread(r'C:\Users\20204916\Documents\GitHub\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\visible\I00041.jpg')
bb=YoloJoeHeller()
cc = bb.detect(aa)
print(cc)