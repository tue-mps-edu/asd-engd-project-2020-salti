#desired height and weight for resizing the image
desired_height=400
desired_width=600

dir_dataset = r"C:\Github\asd-pdeng-project-2020-developer\src experimental\Data\Dataset_V0\images\set00\V000"
dir_sub_thermal = r'\\thermal\\'
dir_sub_rgb = r"\\visible\\"
dir_classes = r'Yolo_config/coco.names'
image_format = ".jpg"

class YoloConfig:
    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3  # lower ==> less number of boxes
    dir_classes = 'Yolo_config/coco.names'
    dir_cfg = 'Yolo_config/yolov3.cfg'
    dir_weights = 'Yolo_config/yolov3.weights'

yolo_cfg = YoloConfig()
