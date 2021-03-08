import pandas as pd
from utils_thermal.utils import *
from pascal_voc_writer import Writer
import cv2
import numpy as np
import math



class Merger():
    def __init__(self,output_path,merged_img_path):
        self.output_path=output_path                  #output path
        self.merged_img_path=merged_img_path
        filename = merged_img_path.split('\\')
        filename = filename[-1].split('.')
        self.filename = filename[0]


    def save_to_format(df,merg_img,self):

        # df = Dataframe storing results of the merged bounding boxes and class names
        # merged_img_path = The path of the merged image. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg'
        # merg_img = The image of interest (In this case: Merged image of the SALTI Program)
        # output_path = Location where pascal voc file needs to be stored. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\output'
        output_path=self.output_path
        img_sz = np.shape(merg_img)  # Get the size of the image

        # Writer(path, width, height) -  adds description about the shape of the image
        writer = Writer(self.merged_img_path, img_sz[0], img_sz[1], img_sz[2])

        # For loop to iterate over all the rows of the Dataframe
        for index, row in df.iterrows():
            #::addObject(name, xmin, ymin, xmax, ymax) - Converts information from Dataframe to pascal voc format
            writer.addObject(df.iloc[index, 6], int(df.iloc[index, 2] - df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] - df.iloc[index, 5] / 2), int(df.iloc[index, 2] + df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] + df.iloc[index, 5] / 2))


        # Save the file to output folder
        writer.save(output_path + '\\' + self.filename + '.xml')

    def save_objects(self, bboxs, confs, classIds, classNames, desired_width, desired_height):

        # For GUI
        CL_GUI = ["Category", "xc", "yc", "w", "h"]
        df_GUI = pd.DataFrame(columns=CL_GUI)

        # Preparing a blank dataframe for each picture's results
        CL = ["Image", "Box", "xc", "yc", "w", "h", "Category", "Confidence"]
        df = pd.DataFrame(columns=CL)
        j = 0
        for i in range(len(bboxs)):
            box, conf, name = bboxs[i], confs[i], classIds[i]
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[
                3]  # Bounding box is X_topleft,Y_topleft while we need X_cent, Y_cent for GUI

            # Storing each picture's results in its dataframe
            df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
            df.at[j, CL[0]] = self.filename
            df.at[j, CL[1]] = j + 1
            df.at[j, CL[2]] = x  # X_centroid for GUI
            df.at[j, CL[3]] = y  # y_centroid for GUI
            df.at[j, CL[4]] = w  # horizontal distance
            df.at[j, CL[5]] = h  # Vertical distance
            df.at[j, CL[6]] = classNames[classIds[i]]
            df.at[j, CL[7]] = confs[i]
            j += 1

            # For GUI
            df_GUI.at[j, CL_GUI[0]] = name
            df_GUI.at[j, CL_GUI[1]] = x / desired_width  # X_centroid for GUI
            df_GUI.at[j, CL_GUI[2]] = y / desired_height  # y_centroid for GUI
            df_GUI.at[j, CL_GUI[3]] = w / desired_width  # horizontal distance
            df_GUI.at[j, CL_GUI[4]] = h / desired_height  # Vertical distance

        # Exporting each picture's results to its specific csv file
        #df.to_csv(os.path.join(path, file_name + '.csv'), index=False)
        # Saving to gui readable format
        df_GUI.to_csv(os.path.join(self.path, self.filename + '.txt'), header=None, index=None, sep=' ')
        df_GUI.to_csv(os.path.join(self.path, self.filename + '_VAL.txt'), header=None, index=None, sep=' ')

        return df
