
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

def get_classes(classes_file_dir):
    classNames = []
    with open(classes_file_dir, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
        classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")
    return classNames

def get_objects(outputs, img_rgb, classNames, cfg):
    hT, wT, cT = img_rgb.shape
    bboxs = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > cfg.confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)-h/2)
                bboxs.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    return bboxs, classIds, confs

def filter_objects(bboxs, confs, classIds, cfg):
    # Non maximum suppression, will give indices to keep
    indices = cv2.dnn.NMSBoxes(bboxs,confs,cfg.confThreshold,cfg.nmsThreshold)
    to_filter = [i[0] for i in indices]
    bboxs_fil, confs_fil, classIds_fil = [], [], []
    if (len(indices)==0):
        return bboxs_fil, confs_fil, classIds_fil
    #assert(len(indices)!=0)
    for i in range(len(bboxs)):
        if i in to_filter:
            bboxs_fil.append(bboxs[i])
            confs_fil.append(confs[i])
            classIds_fil.append(classIds[i])

    return bboxs_fil, confs_fil, classIds_fil

def draw_bboxs(img, bboxs, confs, classIds, classNames):

    for i in range(len(bboxs)):
        box, conf, name = bboxs[i], confs[i], classIds[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

def save_objects(path, file_name, file_ext, bboxs, confs, classIds, classNames):

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
        df_GUI = df_GUI.append(pd.Series([classIds[i],x,y,w,h], index=df_GUI.columns), ignore_index=True)

    #Exporting each picture's results to its specific csv file
    df.to_csv(os.path.join(path, file_name + '.csv'), index=False)
    # df.to_csv()


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

#function to take an input image, resize it according to the desired height and width and save it to a specific directory
def resize_and_save_image(path,path_resized,file,desired_width,desired_height):
    original_image = Image.open(os.path.join(path,file))
    resized_image = original_image.resize((desired_width,desired_height), Image.ANTIALIAS)
    resized_image.save(os.path.join(path_resized,file))

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


