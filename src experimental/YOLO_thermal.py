import cv2
import numpy as np

import pandas as pd
from localFunctions import find_digits, find_objects_and_write, read_and_display_boxes, resize_image,create_archive

print('Test for github')
print("Berend can also run this!")

# Image numbers
start = 0
end = start+200
steps = 50

#desired height and weight for resizing the image
desired_height=400
desired_width=600

#Creating an archive df
df_whole=create_archive('Yolo_config/coco.names')

# Look into folders and choose
dataset = "V0"  # V0 - day, V1 - night
subdataset = "V000" # choose subdataset folder

file_type = [".jpg",".csv"]
i = start
while i <= end:
    d4,d3,d2,d1 = find_digits(i)
    path_rgb = "Data/Dataset_"+dataset+"/images/set00/"+subdataset+"/visible/I0"+str(d4)+str(d3)+str(d2)+str(d1)
    path_thermal = "Data/Dataset_" + dataset + "/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(d2) + str(d1)
    i += steps

    #Giving as input original images to cv2
    # img_rgb = cv2.imread(path_rgb+file_type[0])
    # img_thermal = cv2.imread(path_thermal+file_type[0])

    #Giving a path for saving resized rgb and thermal images with a directory similar to original ones but in a _resized folder
    path_rgb_resized = "Data/Dataset_" + dataset + "_resized/images/set00/" + subdataset + "/visible/I0" + str(d4) + str(d3) + str(d2) + str(
        d1) + "_resized"
    path_thermal_resized = "Data/Dataset_" + dataset + "_resized/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(
        d2) + str(d1) + "_resized"
    #Calling the resize_image functions that takes the original images' paths and file type and the desired width and height
    #and path for saving the resized pictures
    resize_image(path_rgb,file_type[0],desired_width,desired_height,path_rgb_resized)
    resize_image(path_thermal,file_type[0],desired_width,desired_height,path_thermal_resized)
    #giving as input resized images to cv2
    img_rgb = cv2.imread(path_rgb_resized+file_type[0])
    img_thermal = cv2.imread(path_thermal_resized+file_type[0])

    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3  # lower ==> less number of boxes

    classesFile = 'Yolo_config/coco.names'

    classNames = []
    with open(classesFile, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
        classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")

    modelConfiguration = 'Yolo_config/yolov3.cfg'
    modelWeights = 'Yolo_config/yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(img_rgb, 1/255,(whT, whT), [0,0,0],1,False)
  
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    df=find_objects_and_write(outputs, img_rgb, img_thermal, classNames, confThreshold, nmsThreshold, path_thermal, file_type[1],df_whole)

    find_objects_and_write(outputs, img_rgb, img_thermal, classNames, confThreshold, nmsThreshold, path_thermal_resized,
                           file_type[1],df_whole)

    df_whole.append(df, ignore_index=True)

    cv2.imshow("Color ", img_rgb)
    cv2.imshow("Thermal (corresponding) ", img_thermal)
    cv2.waitKey(100) #miliseconds of pause between different pictures

    #read_and_display_boxes(path_thermal)

df_whole.to_csv('Archive.csv')
