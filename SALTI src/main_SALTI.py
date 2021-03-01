import argparse
from config import *
from functions import *
import detect_rgb as netrgb
import detect_thermal as nettherm
import tf

def get_name_ext(filename):
    return os.path.splitext(filename)[0], os.path.splitext(filename)[1]

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
def label_single():
    # TEST RGB YOLO
    dir_test_image = r"Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg"
    img = cv2.imread(dir_test_image)
    net_RGB, classnames_RGB = netrgb.initialize()
    det_RGB, img_RGB = netrgb.detect(net_RGB, classnames_RGB, img)
    cv2.imshow("RGB YOLO", img_RGB)


    # TEST THERMAL YOLO
    # change image from ndarray D=3 to tensor D=3.
    net_T, classnames_T, opt = nettherm.initialize()
    #img_tens = tf.convert_to_tensor(img)
    nettherm.detect(net_T, img, opt)

    print("Insert thermal here")

    # TEST MERGING
    print("Insert merging")

    # Wait until finished
    cv2.waitKey(1000)
    input("Press Enter to finish test...")

label_single()

