import numpy as np
from config import *
from localFunctions import *
import pandas as pd
import os


file_path = "Data\Dataset_V0\images\set00\V000\Validation"
filename_yolo = "I02199_yolo"
filename_gui = "I02199_gui"

def UserValidation(file_path, filename_yolo, filename_gui):
    #this is code for reading output files from YOlO and from the GUI
    #GUI

    a = pd.read_csv(os.path.join(file_path,filename_gui + ".txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
    #a['filename'] = os.path.basename(file_path + "/" + filename_gui + ".txt")
    a['filename'] = filename_gui
    print(a)

    #Yolo
    b = pd.read_csv(os.path.join(file_path,filename_yolo + ".txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
    b['filename'] = filename_yolo
    print(b)


    # maths:
    #print(a.loc[2][1])
    #print(a['x_c'][0])


    for index_a in range(a.shape[0]): #0 to 2 (along columns of gui)
        distances = np.zeros(b.shape[0])
        for index_b in range(b.shape[0]): # 0 to 3 (along columns of yolo)
            #centroid_distances = np.add(a['x_c'][index_a], b['x_c'][index_b])
            centroid_distances = np.sqrt( (a['x_c'][index_a]-b['x_c'][index_b])**2 + (a['y_c'][index_a]-b['y_c'][index_b])**2)
            distances[index_b] = centroid_distances

        print('Distances of a {} from b {} is {}'.format(index_a,index_b,distances))


        #find the minimum
        #minElement,index = np.amin(distances)
        #print(f'this is the min distance between first GUI and all YOLO {minElement}')

        #compare the minimum with diameter of the larger box
        box1 = np.sqrt( (a['w'][index_a]) **2 + (a['h'][index_a]) **2 )
        #print(f'this is distance of the GUI box {box1}')
        box2 = np.sqrt( (b['w'][index_b]) **2 + (b['h'][index_b]) **2 )
        #print(f'this is distance of the YOLO box {box2}')
        #boddim = np.amax(box1, box2)

        #check if the same class or not
        ...


UserValidation(dir_thermal_resized,'I00799_YOLO','I00799')


#read_and_display_boxes(dir_Validation, 'I02199')

