import cv2
from config import *

#from localFunctions import find_digits, find_objects_and_write, read_and_display_boxes, resize_image,create_archive
from localFunctions import *

# Image numbers
start = 0
end = start+200
steps = 50

# Iterate over directory
# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
import os

def label_all_in_folder():
    # Define the network
    net = cv2.dnn.readNetFromDarknet(yolo_cfg.dir_cfg,yolo_cfg.dir_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Create structure for saving data
    objects_classlist = ["Image","Box", "xc", "yc", "w", "h", "Category", "Confidence"]
    objects_dataframe = pd.DataFrame(columns=objects_classlist)

    # Loop over viles in the RGB directory
    for filename in os.listdir(dir_rgb):
        # Skip everything that is the wrong format
        if not filename.endswith(image_format):
            continue
        # Label the rest of the files
        file_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]
        print('dir= '+dir_rgb+', name= '+file_name+', ext= '+file_ext)

        # Read the original images
        #img_rgb = cv2.imread(os.path.join(dir_rgb,file_name+file_ext))
        #img_thermal = cv2.imread(os.path.join(dir_thermal,file_name+file_ext))

        # Resize the images
        resize_and_save_image(dir_rgb,dir_rgb_resized,filename,desired_width,desired_height)
        resize_and_save_image(dir_thermal,dir_thermal_resized,filename,desired_width,desired_height)

        # read resized images
        img_rgb = cv2.imread(os.path.join(dir_rgb_resized,file_name+file_ext))
        img_thermal = cv2.imread(os.path.join(dir_thermal_resized,file_name+file_ext))

        # Get classes
        classNames = get_classes(yolo_cfg.dir_classes)

        # Create BLOB
        blob = cv2.dnn.blobFromImage(img_rgb, 1/255,(yolo_cfg.whT, yolo_cfg.whT), [0,0,0],1,False)

        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

        outputs = net.forward(outputNames)

        #df_whole=df_whole.append(df, ignore_index=True)


        # Get all objects from the outputs
        bboxs, classIds, confs = get_objects(outputs, img_rgb, classNames, yolo_cfg)
        # Filter out those that are below configured threshold
        bboxs_fil, confs_fil, classIds_fil = filter_objects(bboxs, confs, classIds, yolo_cfg)
        # Add the bboxes to the images
        draw_bboxs(img_rgb, bboxs_fil, confs_fil, classIds_fil, classNames)
        draw_bboxs(img_thermal, bboxs_fil, confs_fil, classIds_fil, classNames)
        # Plot the images
        cv2.imshow("Color ", img_rgb)
        cv2.imshow("Thermal (corresponding) ", img_thermal)
        cv2.waitKey(100) #miliseconds of pause between different pictures

        # Save the objects in csv file
        save_objects(objects_dataframe, objects_classlist, dir_thermal, file_name, file_ext, bboxs_fil, confs_fil, classIds_fil, classNames)

    #objects_dataframe.to_csv('Archive.csv')

label_all_in_folder()

# file_type = [".jpg",".csv"]
# i = start
# while i <= end:
#     d4,d3,d2,d1 = find_digits(i)
#     path_rgb = "Data/Dataset_"+dataset+"/images/set00/"+subdataset+"/visible/I0"+str(d4)+str(d3)+str(d2)+str(d1)
#     dir_thermal = "Data/Dataset_" + dataset + "/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(d2) + str(d1)
#     i += steps
#
#     #Giving as input original images to cv2
#     # img_rgb = cv2.imread(path_rgb+file_type[0])
#     # img_thermal = cv2.imread(dir_thermal+file_type[0])
#
#     #Giving a path for saving resized rgb and thermal images with a directory similar to original ones but in a _resized folder
#     path_rgb_resized = "Data/Dataset_" + dataset + "_resized/images/set00/" + subdataset + "/visible/I0" + str(d4) + str(d3) + str(d2) + str(
#         d1) + "_resized"
#     dir_thermal_resized = "Data/Dataset_" + dataset + "_resized/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(
#         d2) + str(d1) + "_resized"
#     #Calling the resize_image functions that takes the original images' paths and file type and the desired width and height
#     #and path for saving the resized pictures
#     resize_image(path_rgb,file_type[0],desired_width,desired_height,path_rgb_resized)
#     resize_image(dir_thermal,file_type[0],desired_width,desired_height,dir_thermal_resized)
#     #giving as input resized images to cv2
#     img_rgb = cv2.imread(path_rgb_resized+file_type[0])
#     img_thermal = cv2.imread(dir_thermal_resized+file_type[0])
#
#     classesFile = 'Yolo_config/coco.names'
#
#     classNames = []
#     with open(classesFile, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
#         classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")
#
#     modelConfiguration = 'Yolo_config/yolov3.cfg'
#     modelWeights = 'Yolo_config/yolov3.weights'
#
#     net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#
#     blob = cv2.dnn.blobFromImage(img_rgb, 1/255,(whT, whT), [0,0,0],1,False)
#
#     net.setInput(blob)
#
#     layerNames = net.getLayerNames()
#     outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
#
#     outputs = net.forward(outputNames)
#
#     #find_objects_and_write(outputs, img_rgb, img_thermal, classNames, confThreshold, nmsThreshold, dir_thermal, file_type[1])
#
#     df=find_objects_and_write(outputs, img_rgb, img_thermal, classNames, confThreshold, nmsThreshold, dir_thermal_resized,
#                            file_type[1])
#
#     df_whole=df_whole.append(df, ignore_index=True)
#
#     cv2.imshow("Color ", img_rgb)
#     cv2.imshow("Thermal (corresponding) ", img_thermal)
#     cv2.waitKey(100) #miliseconds of pause between different pictures
#
#     #read_and_display_boxes(dir_thermal)
#
# df_whole.to_csv('Archive.csv')
