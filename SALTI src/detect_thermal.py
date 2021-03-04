import argparse
from sys import platform

from models_thermal import *  # set ONNX_EXPORT in models.py
from utils_thermal.datasets import *
from utils_thermal.utils import *

import pandas as pd
import sys

from config import *
from functions import *

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=cfg_T.dir_cfg, help='cfg file path')
    parser.add_argument('--data', type=str, default=cfg_T.dir_classes, help='coco.data file path')
    parser.add_argument('--weights', type=str, default='Yolo_config\yolov3-thermal.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default=r'C:\Github\asd-pdeng-project-2020-developer\SALTI src\Data\mytestdata\RGB', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        print("detect_thermal.py, line 32 update config")
        out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Convert image

        # Initialize model
        model = Darknet(opt.cfg, img_size)

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
        classes = load_classes(parse_data_cfg(opt.data)['names'])
#        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    return model, classes, opt, device

    #return model, classes

def convertImage(img0, device, opt):
    # Padded resize
    img = letterbox(img0, new_shape=opt.img_size)[0]

    # Normalize RGB
    #img = img0
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if opt.half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    img = torch.from_numpy(img).to(device)
    print('original'+str(img0.shape))
    print('resized'+str(img.shape))

    return img, img0

def detect(model, img, opt, device):
    # Run inference
    with torch.no_grad():
        # Get detections
        img, img0 = convertImage(img,device, opt)

        #print("check if image is right format")
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        bboxs, confs, classes = getlists(pred, img, img0)
        #det = Detections(bboxs, classes,confs)
    return bboxs, confs, classes
