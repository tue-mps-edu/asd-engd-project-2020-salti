import cv2
import numpy as np
from localFunctions import find_digits

# Image numbers
start = 0
end = 2244
steps = 150

# Look into folders and choose
dataset = "V1"  # V0 - day, V1 - night
subdataset = "V000" # choose subdataset folder


i = start
while i <= end:
    d4,d3,d2,d1 = find_digits(i)
    path_rgb = "Dataset_"+dataset+"/images/set00/"+subdataset+"/visible/I0"+str(d4)+str(d3)+str(d2)+str(d1)+".jpg"
    path_thermal = "Dataset_" + dataset + "/images/set00/" + subdataset + "/thermal/I0" + str(d4) + str(d3) + str(d2) + str(d1) + ".jpg"
    i += steps

    img_rgb = cv2.imread(path_rgb)
    img_thermal = cv2.imread(path_thermal)

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


    def findObjects(outputs, img_rgb, img_therm):
        hT, wT, cT = img_rgb.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    w,h = int(det[2]*wT), int(det[3]*hT)
                    x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

        for i in indices:
            i = i[0]  # to squeeze the dimension
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(img_rgb,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                        (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

            cv2.rectangle(img_therm, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img_therm, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    blob = cv2.dnn.blobFromImage(img_rgb, 1/255,(whT, whT), [0,0,0],1,False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img_rgb, img_thermal)


    cv2.imshow("Color ", img_rgb)
    cv2.imshow("Thermal (corresponding) ", img_thermal)
    cv2.waitKey(5)