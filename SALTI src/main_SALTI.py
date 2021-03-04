import argparse
from config import *
from functions import *
import detect_rgb as netrgb
import detect_thermal as nettherm
import tensorflow as tf
import torch
from preprocess import *


def get_name_ext(filename):
    return os.path.splitext(filename)[0], os.path.splitext(filename)[1]

def label_single():
    '''
    Test script for labelling a single image
    it runs the RGB yolo deteection, which returns detection for single image
    Then it runs the rgb yolo detection, which returns detection for single image
    :return:
    '''


    #RESIZE THE PICTURES
    #resize_and_save_image(dir_thermal,dir_thermal_resized,"I00000.jpg",416, 256)
    # dir_thermal_test_image = r"Data\Dataset_V0\images\set00\V000\thermal_resized\I00000.jpg"

    # TEST RGB YOLO
    dir_rgb_test_image = r"Data\Dataset_V0\images\set00\V000\visible\I00000.jpg"
    dir_thermal_test_image = r"Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg"

    img_C = cv2.imread(dir_rgb_test_image)
    img_T = cv2.imread(dir_thermal_test_image)
    img_M = img_T.copy()

    net_RGB, classnames_RGB = netrgb.initialize()
    boxes_C, confs_C, classes_C = netrgb.detect(net_RGB, classnames_RGB, img_C)

    # Add Bounding Boxes to image
    draw_bboxs(img_C, boxes_C, confs_C, classes_C, classnames_RGB)
    cv2.imshow("RGB YOLO", img_C)

    # TEST THERMAL YOLO
    net_T, classnames_T, opt, device = nettherm.initialize()
    boxes_T, confs_T, classes_T = nettherm.detect(net_T, img_T, opt, device)

    # Add Bounding Boxes to image
    draw_bboxs(img_T, boxes_T, confs_T, classes_T, classnames_RGB)
    cv2.imshow("THERMAL", img_T)

    #nettherm.detect_old()

    print("Insert thermal here")

    # TEST MERGING
    assert(type(boxes_C)==type(boxes_T))
    boxes, classes, confs = nms(boxes_C + boxes_T, confs_C+confs_T, classes_C+classes_T, cfg_T.confThreshold, cfg_T.nmsThreshold)

    #Exporting the results
    df=save_objects(r"Data\Dataset_V0\images\set00\V000\thermal", "I00000", ".jpg", boxes, confs, classes, classnames_T, 640, 512)

    # Add Bounding Boxes to image
    draw_bboxs(img_M, boxes, confs, classes, classnames_RGB)
    cv2.imshow("MERGED", img_M)

    #(boxes, confidences, classes, conf_threshold=0.5, nms_threshold=0.3 ):

    # det_merged = mergefunction(...)
    # merged labels = nms(...)
    print("Insert merging")

    # Exporting

    # Wait until finished
    # Wait until finished
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    input("Press Enter to finish test...")

label_single()



'''' READING IMAGES
        # Read the original images
        #img_rgb = cv2.imread(os.path.join(dir_rgb,file_name+file_ext))
        #img_thermal = cv2.imread(os.path.join(dir_thermal,file_name+file_ext))

        # Resize the images
        #resize_and_save_image(dir_rgb,dir_rgb_resized,filename,desired_width,desired_height)
        #resize_and_save_image(dir_thermal,dir_thermal_resized,filename,desired_width,desired_height)
        # read resized images
        #img_rgb = cv2.imread(os.path.join(dir_rgb_resized,file_name+file_ext))
        #img_thermal = cv2.imread(os.path.join(dir_thermal_resized,file_name+file_ext))
'''
