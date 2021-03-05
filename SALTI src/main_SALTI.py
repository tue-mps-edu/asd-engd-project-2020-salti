from config import *
from functions import *
import detect_rgb as netrgb
import detect_thermal as nettherm
from preprocess import *


def get_name_ext(filename):
    return os.path.splitext(filename)[0], os.path.splitext(filename)[1]


def label_loop(image_path):

    # Initialize the thermal and RGB YOLO
    net_RGB, classnames_RGB = netrgb.initialize()
    net_T, classnames_T, opt, device = nettherm.initialize()

    i = 0
    for filename_thermal in os.listdir(dir_thermal):
        i = i+1
        if i%50!=0:
            continue

        # Skip everything that is the wrong format
        if not filename_thermal.endswith(image_format):
            continue

        file_name = os.path.splitext(filename_thermal)[0]
        file_ext = os.path.splitext(filename_thermal)[1]

        print('Processing file: '+str(os.path.join(dir_rgb,filename_thermal)))
        print('Processing file: '+str(os.path.join(dir_thermal,filename_thermal)))

        rgb_image_path = os.path.join(dir_rgb,filename_thermal)
        thermal_image_path = os.path.join(dir_thermal,filename_thermal)
        img_C = cv2.imread(rgb_image_path)
        img_T = cv2.imread(thermal_image_path)
        img_M = img_T.copy()

        # Perform detection
        boxes_C, confs_C, classes_C = netrgb.detect(net_RGB, classnames_RGB, img_C)
        boxes_T, confs_T, classes_T = nettherm.detect(net_T, img_T, opt, device)

        # Add Bounding Boxes to image
        draw_bboxs(img_C, boxes_C, confs_C, classes_C, classnames_RGB)
        cv2.imshow("RGB YOLO", img_C)

        # Add Bounding Boxes to image
        draw_bboxs(img_T, boxes_T, confs_T, classes_T, classnames_RGB)
        cv2.imshow("THERMAL", img_T)

        # TEST MERGING
        assert (type(boxes_C) == type(boxes_T))
        boxes, classes, confs = nms(boxes_C + boxes_T, confs_C + confs_T, classes_C + classes_T,
                                    conf_threshold_ensemble, nms_threshold_ensemble)

        # Exporting the results
        df = save_objects(dir_thermal, file_name, file_ext, boxes, confs, classes,classnames_T, output_width, output_height)

        # Add Bounding Boxes to image
        draw_bboxs(img_M, boxes, confs, classes, classnames_RGB)
        cv2.imshow("MERGED", img_M)
        cv2.waitKey(5000)

#label_loop(dir_dataset)

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
    dir_rgb_test_image = r"Data\Dataset_V0\images\set00\V000\visible\I01588.jpg"
    dir_thermal_test_image = r"Data\Dataset_V0\images\set00\V000\thermal\I01588.jpg"

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
    df=save_objects(r"Data\Dataset_V0\images\set00\V000\thermal", "I01588", ".jpg", boxes, confs, classes, classnames_T, 640, 512)
    # read_and_display_boxes(r"Data\Dataset_V0\images\set00\V000\thermal", "I01588")


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
def main():
    #label_single()
    resize_and_save_image(r"D:\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\visible",
        r"D:\asd-pdeng-project-2020-developer\SALTI src\Data", "I00000.jpg", 340, 340)


if __name__ == "__main__":
    main()
