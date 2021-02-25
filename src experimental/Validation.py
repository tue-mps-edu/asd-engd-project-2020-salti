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
    #a['filename'] = os.path.basename(file_path + "/" + filename_gui + ".txt")
    GU['filename'] = filename_gui
    print(GU)

    #Yolo
    YO = pd.read_csv(os.path.join(file_path,filename_yolo + ".txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
    YO['filename'] = filename_yolo
    print(YO)


    # maths:
    #print(a.loc[2][1])
    #print(a['x_c'][0])


    for index_GU in range(GU.shape[0]): #0 to 2 (along columns of gui)
        distances = np.zeros(YO.shape[0])
        for index_YO in range(YO.shape[0]): # 0 to 3 (along columns of yolo)
            #centroid_distances = np.add(a['x_c'][index_a], b['x_c'][index_b])
            centroid_distances = np.sqrt( (GU['x_c'][index_GU]-YO['x_c'][index_YO])**2 + (GU['y_c'][index_GU]-YO['y_c'][index_YO])**2)
            distances[index_YO] = centroid_distances
        #print('Distances of a {} from YO are {}'.format(index_GU,distances))



        #find the minimum
        minElement = np.amin(distances)
        ind = np.argmin(distances)
        print('The min distance between GU {} from YO is {} at index {}'.format(index_GU,minElement,ind))

        #compare the minimum with diameter of the larger box
        box1 = np.sqrt( (GU['w'][index_GU]) **2 + (GU['h'][index_GU]) **2 )
        print(f'this is distance of the GUI box {box1}')
        box2 = np.sqrt( (YO['w'][ind]) **2 + (YO['h'][ind]) **2 )
        #print(f'this is distance of the YOLO box {box2}')
        #boddim = np.amax(box1, box2)

        #check if the same class or not
        ...


UserValidation(dir_thermal_resized,'I00799_YOLO','I00799')


#read_and_display_boxes(dir_Validation, 'I02199')

