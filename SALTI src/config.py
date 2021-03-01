DEBUG_MODE = True

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

class ConfigRGB:
    # Configuration of YOLO color detection algorithm
    whT = 320               # width & height of the image input into YOLO (standard resolution, square)
    confThreshold = 0.5     # Confidence threshold for approval of detection
    nmsThreshold = 0.3      # Non-maximum suppresion threshold (lower = less number)
    dir_classes = 'Yolo_config/coco-rgb.names'
    dir_cfg = 'Yolo_config/yolov3-rgb.cfg'
    dir_weights = 'Yolo_config/yolov3-rgb.weights'

class ConfigThermal:
    # Configuration of YOLO thermal detection algorithm
    whT = 320               # width & height of the image input into YOLO (standard resolution, square)
    confThreshold = 0.5     # Confidence threshold for approval of detection
    nmsThreshold = 0.3      # Non-maximum suppresion threshold (lower = less number)
    #dir_classes = 'Yolo_config\coco-thermal.names'
    dir_classes = 'Yolo_config/coco-thermal.data'
    #dir_classes = 'data/coco.data'
    dir_cfg = 'Yolo_config/yolov3-spp.cfg'
    dir_weights = 'Yolo_config\yolov3-thermal.weights'

cfg_C = ConfigRGB()
cfg_T = ConfigThermal()






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