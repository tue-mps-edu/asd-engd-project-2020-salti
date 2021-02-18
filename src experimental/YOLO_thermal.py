import cv2
import numpy as np
import pandas as pd
from localFunctions import find_digits, find_objects_and_write, read_and_display_boxes

# Image numbers
start = 0
end = 0
steps = 1

# Look into folders and choose
dataset = "V0"  # V0 - day, V1 - night
subdataset = "V000" # choose subdataset folder

file_type = [".jpg",".csv"]
i = start
while i <= end:
    d4,d3,d2,d1 = find_digits(i)
    path_rgb = "Dataset_"+dataset+"/images/set00/"+subdataset+"/visible/I0"+str(d4)+str(d3)+str(d2)+str(d1)
    path_thermal = "Dataset_" + dataset + "/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(d2) + str(d1)
    i += steps

    img_rgb = cv2.imread(path_rgb+file_type[0])
    img_thermal = cv2.imread(path_thermal+file_type[0])

    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3  # lower ==> less number of boxes

    classesFile = 'coco.names'
    classNames = []
    with open(classesFile, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
        classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")

    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(img_rgb, 1/255,(whT, whT), [0,0,0],1,False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    # find_objects_and_write(outputs, img_rgb, img_thermal, classNames, confThreshold, nmsThreshold, path_thermal, file_type[1])
    # cv2.imshow("Color ", img_rgb)
    # cv2.imshow("Thermal (corresponding) ", img_thermal)
    # cv2.waitKey(5)

    read_and_display_boxes(path_thermal)