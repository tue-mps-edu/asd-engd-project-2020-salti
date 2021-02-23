import numpy as np
from config import *
from localFunctions import *

file_path = "Data\Dataset_V0\images\set00\V000\YOLO-GUI"
filename_yolo = "I02199"
filename_gui = "I02199"

def UserValidation(file_path, filname_yolo, filename_gui):
    import pandas as pd
    import os
    #this is code for reading output files from YOlO and from the GUI
    #YOLO

    a = pd.read_csv(file_path + "/" + filname_yolo + ".csv", names=["classes", "x_c", "ÿ_c", "w", "h"])
    print(a)

    #GUI

    b = pd.read_csv(file_path + "/" + filename_gui + ".txt", delim_whitespace=True,
                    names=["classes", "x_c", "ÿ_c", "w", "h"])
    b['filename'] = os.path.basename(file_path + "/"+ filename_gui + ".txt")
    print(b)

    # maths:
    #print(b.values)


UserValidation(file_path, filename_yolo,filename_gui)

#read_and_display_boxes(dir_thermal_resized,"I00199")
