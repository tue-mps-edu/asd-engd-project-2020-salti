
import numpy as np
import cv2
import pandas as pd
from PIL import Image

def find_digits(i):
    d1 = i
    d2 = 0
    d3 = 0
    d4 = 0
    if i >= 10:
        d1 = i % 10
        d2 = int((i - d1) / 10)
        d3 = 0
        d4 = 0
        if i >= 100:
            d1 = i % 10
            d2 = int(((i - d1) / 10) % 10)
            d3 = int((((i - d1) / 10) - d2) / 10)
            d4 = 0
            if i >= 1000:
                d1 = i % 10
                d2 = int(((i - d1) / 10) % 10)
                d3 = int(((((i - d1) / 10) - d2) / 10) % 10)
                d4 = int((((((i - d1) / 10) - d2) / 10) - d3) / 10)

    return d4,d3,d2,d1


def find_objects_and_write(outputs, img_rgb, img_therm, classNames, confThreshold, nmsThreshold, path_thermal, file_type, df_whole):
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

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)  # Non maximum suppression, will give indices to keep

    CL = classNames.copy()
    CL.insert(0, "Box")
    CL.insert(1, "xc")
    CL.insert(2, "yc")
    CL.insert(3, "w")
    CL.insert(4, "h")
    df = pd.DataFrame(columns=CL)
    j = 0

    #Under development for creating a global csv file containing all the images' results
    # CL_whole=classNames.copy()
    # CL_whole.insert(0, "Image")
    # CL_whole.insert(1, "Box")
    # CL_whole.insert(2, "xc")
    # CL_whole.insert(3, "yc")
    # CL_whole.insert(4, "w")
    # CL_whole.insert(5, "h")
    # df_whole=pd.DataFrame(columns=CL_whole)
    # jj=0


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

        df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
        df[CL[0]][j] = j+1
        df[CL[1]][j] = x
        df[CL[2]][j] = y
        df[CL[3]][j] = w
        df[CL[4]][j] = h
        df[classNames[classIds[i]]][j] = confs[i]
        j += 1

        #under development
        # archive_results(path_thermal[-4:],jj,x,y,w,h,confs[i])
        # df_whole = df_whole.append(pd.Series(0, index=df_whole.columns), ignore_index=True)
        # df_whole[CL_whole[0]][j] = path_thermal[-4:]
        # df_whole[CL_whole[1]][j] = jj+1
        # df_whole[CL_whole[2]][j] = x
        # df_whole[CL_whole[3]][j] = y
        # df_whole[CL_whole[4]][j] = w
        # df_whole[CL_whole[5]][j] = h
        # df_whole[classNames[classIds[i]]][j] = confs[i]
        # jj += 1


    df.to_csv(path_thermal + file_type, index=False)


    #TEST
    return df
    # df_whole.append(df,ignore_index=True)
    # df.to_csv('Archive.csv')



def read_and_display_boxes(file_path):
    df = pd.read_csv(file_path+".csv")
    img_thermal = cv2.imread(file_path + ".jpg")
    for i in df.index:
        x, y, w, h = df['xc'][i], df['yc'][i], df['w'][i], df['h'][i]
        df_class = df.drop(['Box','xc','yc','w','h'], axis = 1)
        className = df_class.idxmax(axis=1)[i]
        cv2.rectangle(img_thermal, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img_thermal, f'{className} {int(df[className][i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.imshow("Thermal image with boxes ", img_thermal)
    cv2.waitKey(0)

#function to take an input image, resize it according to the desired height and width and save it to a specific directory
def resize_image(path,file_type,desired_width,desired_height,path_resized):
    original_image = Image.open(path+file_type)
    resized_image = original_image.resize((desired_width,desired_height), Image.ANTIALIAS)
    resized_image.save(path_resized+file_type)
    return resized_image

#Function to save all the results in one file
def create_archive(classesFileName):
    classesFile = classesFileName #'coco.names'
    classNames = []
    with open(classesFile, 'r') as f:                       # using WITH function takes away the need to use CLOSE file function
        classNames = f.read().rstrip('\n').split('\n')      # rstrip strips off the ("content here") and split splits off for("content here")
    CL_whole = classNames.copy()
    CL_whole.insert(0, "Image")
    CL_whole.insert(1, "Box")
    CL_whole.insert(2, "xc")
    CL_whole.insert(3, "yc")
    CL_whole.insert(4, "w")
    CL_whole.insert(5, "h")
    df_whole = pd.DataFrame(columns=CL_whole)
    return df_whole

    # if Archive == True:
    #     CL_whole = classNames.copy()
    #     CL_whole.insert(0, "Image")
    #     CL_whole.insert(1, "Box")
    #     CL_whole.insert(2, "xc")
    #     CL_whole.insert(3, "yc")
    #     CL_whole.insert(4, "w")
    #     CL_whole.insert(5, "h")
    #     df_whole = pd.DataFrame(columns=CL_whole)
    #     return df_whole

# def archive_results(df_whole,df):
#     #df_whole = df_whole.append(pd.Series(0, index=df_whole.columns), ignore_index=True)
#     df_whole.append(df,ignore_index=True)


    # df_whole[CL_whole[0]][j] = path_thermal[-4:]
    # df_whole[CL_whole[1]][j] = jj+1
    # df_whole[CL_whole[2]][j] = x
    # df_whole[CL_whole[3]][j] = y
    # df_whole[CL_whole[4]][j] = w
    # df_whole[CL_whole[5]][j] = h
    # df_whole[classNames[classIds[i]]][j] = confs[i]
    # jj += 1
