import argparse
from config import *
from functions import *
import detect_rgb as netrgb

def labeldir(save_txt=False, save_img=False):
    '''
    This is the main function for the labeling program
    '''

    print("Input the main code here!")

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
def test_detection():
    # TEST RGB YOLO
    dir_test_image = r"C:\Github\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg"
    img = cv2.imread(dir_test_image)
    net, classnames = netrgb.initialize()
    det_RGB, img_RGB = netrgb.detect(net, classnames, img)
    cv2.imshow("RGB YOLO", img_RGB)

    # TEST THERMAL YOLO
    print("Insert thermal here")

    # TEST MERGING
    print("Insert merging")

    # Wait until finished
    cv2.waitKey(1000)
    input("Press Enter to finish test...")

test_detection()

# def detect():
#     # Iterate over directory
#     # https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
#     net_color = netrgb.initialize()
#
#     i = 0
#     for filename in os.listdir(dir_rgb):
#         # For skipping images
#         i = i + 1
#         if i%200!=0:
#             continue
#         # Skip everything that is the wrong format
#         if not filename.endswith(image_format):
#             continue
#         # Split the file name and extension
#         file_name, file_ext = get_name_ext(filename)
#         print('dir= '+dir_rgb+', name= '+file_name+', ext= '+file_ext)
#
#         # DO DETECTION HERE
#         '''' DO DETECTION HERE '''
#         outputs
#
#
#         # Get all objects from the outputs
#         bboxs, classIds, confs = get_objects(outputs, img_rgb, classNames, yolo_cfg)
#         # Filter out those that are below configured threshold
#         bboxs_fil, confs_fil, classIds_fil = filter_objects(bboxs, confs, classIds, yolo_cfg)
#
#         # Add the bboxes to the images
#         draw_bboxs(img_rgb, bboxs_fil, confs_fil, classIds_fil, classNames)
#         draw_bboxs(img_thermal, bboxs_fil, confs_fil, classIds_fil, classNames)
#
#         show_images(200)
#
#         '''' SAVE OBJECTS HERE '''
#         df = save_objects(dir_thermal_resized, file_name, file_ext, bboxs_fil, confs_fil, classIds_fil, classNames,desired_width,desired_height)
#         df_whole=df_whole.append(df, ignore_index=True)






# if __name__ == '__main__':
#     '''
#     This can be used for running the function with different arguments.
#     '''
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
#     parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
#     parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
#     parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     opt = parser.parse_args()
#     print(opt)
#
#     labeldir()
