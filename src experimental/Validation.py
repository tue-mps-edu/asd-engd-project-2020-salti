import numpy as np
from config import *
from localFunctions import *
import pandas as pd
import os


def UserValidation(file_path, filename_yolo, filename_gui):
    #this is code for reading output files from YOlO and from the GUI
    #GUI
    GU = pd.read_csv(os.path.join(file_path,filename_gui + ".txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
    GU['filename'] = filename_gui

    #Yolo
    YO = pd.read_csv(os.path.join(file_path,filename_yolo + ".txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
    YO['filename'] = filename_yolo

    TP=0
    FP=0
    FN=0
    distances_mat=np.zeros((GU.shape[0],YO.shape[0]))

    for index_GU in range(GU.shape[0]): #0 to 2 (along columns of gui)
        for index_YO in range(YO.shape[0]): # 0 to 3 (along columns of yolo)
            centroid_distances = np.sqrt( (GU['x_c'][index_GU]-YO['x_c'][index_YO])**2 + (GU['y_c'][index_GU]-YO['y_c'][index_YO])**2)
            distances_mat[index_GU][index_YO] = centroid_distances
    #print('Distances matrix of size {} is: \n {}'.format(distances_mat.shape,distances_mat))

    #Creating a dataframe for distances between GUI (in rows) boxes and Yolo (in column) boxes
    distances_df=pd.DataFrame(data=distances_mat)

    skip_counter=0
    while distances_df.shape[0] >= 1+skip_counter and distances_df.shape[1] >= 1+skip_counter:
        # To find the indexes of global minimum inside the dataframe
        GUI_min_ind = distances_df.min(axis=1).idxmin()
        YO_min_ind = distances_df.min().idxmin()


        #compare the minimum with diameter of the larger box
        GUI_diam = np.sqrt( (GU['w'][GUI_min_ind]) **2 + (GU['h'][GUI_min_ind]) **2 )
        YO_diam = np.sqrt((YO['w'][YO_min_ind]) ** 2 + (YO['h'][YO_min_ind]) ** 2)
        bigger_diameter = max(GUI_diam, YO_diam)
        if distances_df[GUI_min_ind][YO_min_ind] < bigger_diameter:
            #They are the same boxes
            if GU['classes'][GUI_min_ind]==YO['classes'][YO_min_ind]:
                TP+=1
            else:
                FP+=1

            # To discard the analyzed row and column
            distances_df = distances_df.drop(index=[GUI_min_ind], columns=[YO_min_ind])
        else:
            skip_counter+=1


    FN+=distances_df.shape[0]
    FP+=distances_df.shape[1]

    print('True positive = {}'.format(TP))
    print('False positive = {}'.format(FP))
    print('False Negative = {}'.format(FN))
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = TP / (TP + FP + FN)
    print('Precision is {}, Recall is {} and Accuracy is {}'.format(Precision, Recall, Accuracy))

UserValidation(dir_thermal_resized,'I00799_YOLO','I00799')


#read_and_display_boxes(dir_Validation, 'I02199')

