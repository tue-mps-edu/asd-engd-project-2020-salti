import argparse
from config import *
from functions import *
import detect_rgb as netrgb
import detect_thermal as nettherm
import tensorflow as tf
import torch


def get_name_ext(filename):
    return os.path.splitext(filename)[0], os.path.splitext(filename)[1]

def label_single():
    '''
    Test script for labelling a single image
    it runs the RGB yolo deteection, which returns detection for single image
    Then it runs the rgb yolo detection, which returns detection for single image
    :return:
    '''
    # TEST RGB YOLO
    dir_test_image = r"Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg"
    img = cv2.imread(dir_test_image)
    net_RGB, classnames_RGB = netrgb.initialize()
    #det_RGB, img_RGB = netrgb.detect(net_RGB, classnames_RGB, img)
    #cv2.imshow("RGB YOLO", img_RGB)


    # TEST THERMAL YOLO
    net_T, classnames_T, opt, device = nettherm.initialize()

    # change image from ndarray D=3 to tensor D=3.
    #img_T = torch.from_numpy(img.to(device))
    #img_tens = tf.convert_to_tensor(img)
    #img_tens = tf.image.convert_image_dtype(img, dtype=tf.float16, saturate=False)
    det_T = nettherm.detect(net_T, img, opt, device)
    #nettherm.detect_old()

    print("Insert thermal here")

    # TEST MERGING
    # det_merged = mergefunction(...)
    # merged labels = nms(...)
    print("Insert merging")

    # Exporting

    # Wait until finished
    cv2.waitKey(1000)
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
