
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt


'''
Class for storing all detections
'''
import cv2
import numpy as np


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



def draw_bboxs(img, bboxs, confs, classIds, classNames):

    for i in range(len(bboxs)):
        box, conf, name = bboxs[i], confs[i], classIds[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
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
        x,y,w,h = box[0],box[1],box[2],box[3]

        #Storing each picture's results in its dataframe
        df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
        df.at[j,CL[0]] = os.path.basename(path+file_name+file_ext)
        df.at[j,CL[1]] = j+1
        df.at[j,CL[2]] = x
        df.at[j,CL[3]] = y
        df.at[j,CL[4]] = w
        df.at[j,CL[5]] = h
        df.at[j,CL[6]]=classNames[classIds[i]]
        df.at[j,CL[7]]=confs[i]
        j += 1

        #For GUI
        df_GUI = df_GUI.append(pd.Series([classIds[i],x/desired_width,y/desired_height,w/desired_width,h/desired_height], index=df_GUI.columns), ignore_index=True)

    #Exporting each picture's results to its specific csv file
    df.to_csv(os.path.join(path, file_name + '.csv'), index=False)
    #Saving to gui readable format
    df_GUI.to_csv(os.path.join(path,file_name+'.txt'), header=None, index=None, sep=' ')
    df_GUI.to_csv(os.path.join(path, file_name + '_YOLO.txt'), header=None, index=None, sep=' ')

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


