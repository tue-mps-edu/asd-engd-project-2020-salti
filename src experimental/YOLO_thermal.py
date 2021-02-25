import cv2
from config import *

#from localFunctions import find_digits, find_objects_and_write, read_and_display_boxes, resize_image,create_archive
from localFunctions import *

# Iterate over directory
# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
import os

def label_all_in_folder():
    # Define the network
    net = cv2.dnn.readNetFromDarknet(yolo_cfg.dir_cfg,yolo_cfg.dir_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    df_whole = create_archive()

    # Loop over viles in the RGB directory
    i = 0
    for filename in os.listdir(dir_rgb):
        i = i + 1
        if i%200!=0:
            continue
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
        df = save_objects(dir_thermal_resized, file_name, file_ext, bboxs_fil, confs_fil, classIds_fil, classNames,desired_width,desired_height)
        df_whole=df_whole.append(df, ignore_index=True)

    df_whole.to_csv(os.path.join(dir_PostAnalysis, 'Archive.csv'))
    analyze_results(df_whole,dir_PostAnalysis)



label_all_in_folder()


