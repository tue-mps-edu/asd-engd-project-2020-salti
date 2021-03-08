
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from utils.utils import *

'''
Class for storing all detections
'''
import cv2
import numpy as np
import math


class Detections():
    def __init__(self, boundingBoxes, classes, confidences):
        self.bboxs = boundingBoxes
        self.classIds = classes
        self.confs = confidences

# def nms(boxes, confidences, classes, conf_threshold=0.5, nms_threshold=0.3 ):
#     # Non maximum suppression, will give indices to keep
#     indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
#     indices = indices.reshape((-1,))
#     return boxes[indices], confidences[indices], classes[indices]

def nms(boxes, confidences, classes, conf_threshold=0.5, nms_threshold=0.3 ):
    # Non maximum suppression, will give indices to keep
    indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
    to_keep = [i[0] for i in indices]
    return [boxes[i] for i in to_keep], [classes[i] for i in to_keep], [confidences[i] for i in to_keep]

# def getlists(preds):
#     bboxs, confs, classes = [], [], []
#     for i, det in enumerate(preds):
#         for *xyxy, conf, _, cls in det:
#             t = np.squeeze(xyxy)
#             bboxs.append([int((t[0]+t[2])/2),int((t[1]+t[3])/2),int(abs(t[0]-t[2])),int(abs(t[1]-t[3]))])
#             confs.append(float(conf))
#             classes.append(int(cls))
#     return bboxs, confs, classes

def getlists(preds, img, im0):
    bboxs, confs, classes = [], [], []
    for i, det in enumerate(preds):
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, _, cls in det:  # Save the bounding box coordinates [Top left coordinates, width, height]
                t = np.squeeze(xyxy) #xyxy is x_topLeft,y_topLeft,x_bottomRight,y_bottomRight
                bboxs.append([int(t[0]),int(t[1]),int(abs(t[0]-t[2])),int(abs(t[1]-t[3]))])
                confs.append(float(conf)) # Confidence factor
                classes.append(int(cls))  # Class names
    return bboxs, confs, classes



def draw_bboxs(img, bboxs, confs, classIds, classNames):

    for i in range(len(bboxs)):
        box, conf, name = bboxs[i], confs[i], classIds[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2) #rectangle(starting point which is top left, ending point which is bottom right, color , thickness)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

def save_objects(path, file_name, file_ext, bboxs, confs, classIds, classNames,desired_width,desired_height):

    #For GUI
    CL_GUI=["Category","xc","yc","w","h"]
    df_GUI=pd.DataFrame(columns=CL_GUI)

    #Preparing a blank dataframe for each picture's results
    CL = ["Image","Box", "xc", "yc", "w", "h", "Category", "Confidence"]
    df = pd.DataFrame(columns=CL)
    j = 0
    for i in range(len(bboxs)):
        box, conf, name = bboxs[i], confs[i], classIds[i]
        x,y,w,h = int(box[0]+box[2]/2),int(box[1]+box[3]/2),box[2],box[3] #Bounding box is X_topleft,Y_topleft while we need X_cent, Y_cent for GUI

        #Storing each picture's results in its dataframe
        df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
        df.at[j,CL[0]] = os.path.basename(path+file_name+file_ext)
        df.at[j,CL[1]] = j+1
        df.at[j,CL[2]] = x #X_centroid for GUI
        df.at[j,CL[3]] = y #y_centroid for GUI
        df.at[j,CL[4]] = w #horizontal distance
        df.at[j,CL[5]] = h #Vertical distance
        df.at[j,CL[6]] = classNames[classIds[i]]
        df.at[j,CL[7]] = confs[i]
        j += 1

        #For GUI
        df_GUI.at[j, CL_GUI[0]] = name
        df_GUI.at[j, CL_GUI[1]] = x / desired_width  # X_centroid for GUI
        df_GUI.at[j, CL_GUI[2]] = y / desired_height  # y_centroid for GUI
        df_GUI.at[j, CL_GUI[3]] = w / desired_width  # horizontal distance
        df_GUI.at[j, CL_GUI[4]] = h / desired_height  # Vertical distance

    #Exporting each picture's results to its specific csv file
    df.to_csv(os.path.join(path, file_name + '.csv'), index=False)
    #Saving to gui readable format
    df_GUI.to_csv(os.path.join(path,file_name+'.txt'), header=None, index=None, sep=' ')
    df_GUI.to_csv(os.path.join(path, file_name + '_VAL.txt'), header=None, index=None, sep=' ')

    return df


def read_and_display_boxes(file_path, file):
    df = pd.read_csv(os.path.join(file_path, file + ".csv"))
    img_thermal = cv2.imread(os.path.join(file_path, file + ".jpg"))
    for i in df.index:
        x, y, w, h, className, confidence = df['xc'][i], df['yc'][i], df['w'][i], df['h'][i], df['Category'][i],df['Confidence'][i]
        cv2.rectangle(img_thermal, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img_thermal, f'{className} {int(confidence * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.imshow("Thermal image with boxes ", img_thermal)
    cv2.waitKey(0)



#Function to create a data frame that saves all the results in one file
def create_archive():
    CL = ["Image", "Box", "xc", "yc", "w", "h", "Category", "Confidence"]
    df_whole = pd.DataFrame(columns=CL)
    return df_whole

#Function to give insights about the results
def analyze_results(df_whole,path):
    plt.figure()
    plt.hist(df_whole['Category'],)
    plt.ylabel('Number of detections')
    plt.xticks(df_whole.Category.unique(),rotation='vertical')
    #plt.savefig('Histogram of Labelled Categories')
    plt.savefig(os.path.join(path,'Cats'))

    plt.figure()
    plt.hist(df_whole['Confidence'])
    plt.ylabel('Number of detections')
    plt.xlabel('Confidence level')
    #plt.savefig('Histogram of Confidence levels')
    plt.savefig(os.path.join(path,'Confidence levels'))
    #plt.figure()
    #plt.show()

    labels=df_whole.Category.unique()
    for item in labels:
        confidence_per_item=pd.Series(df_whole['Confidence'][df_whole['Category']==item])
        plt.figure()
        plt.hist(confidence_per_item)
        plt.ylabel('Detections')
        plt.title(item)
        plt.savefig(os.path.join(path,item))


