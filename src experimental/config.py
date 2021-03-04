#desired height and weight for resizing the image
desired_height=400
desired_width=600
import os, errno

# Define image locations
dir_dataset = os.path.join(os.getcwd(),'Data\Dataset_V0\images\set00\V000')
dir_thermal = os.path.join(dir_dataset,'thermal')
dir_rgb = os.path.join(dir_dataset,'visible')
dir_thermal_resized = os.path.join(dir_dataset,'thermal_resized')
dir_rgb_resized = os.path.join(dir_dataset,'rgb_resized')
dir_classes = os.path.join(os.getcwd(),'Yolo_config\coco.names')
dir_PostAnalysis=os.path.join(os.getcwd(),'Post_Analysis')
dir_Validation=os.path.join(dir_dataset,'Validation')

image_format = ".jpg"

class YoloConfigRGB:
    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3  # lower ==> less number of boxes
    dir_classes = 'Yolo_config/coco.names'
    dir_cfg = 'Yolo_config/yolov3.cfg'
    dir_weights = 'Yolo_config/yolov3.weights'

yolo_cfg = YoloConfigRGB()

# Create folders for resized images
try:
    os.mkdir(dir_thermal_resized)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
try:
    os.mkdir(dir_rgb_resized)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

#Create folder for post analysis results
try:
    os.mkdir(dir_PostAnalysis)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

#Create
try:
    os.mkdir(dir_Validation)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
