import pandas as pd
from utils_thermal.utils import *
from pascal_voc_writer import Writer
import cv2
import numpy as np
import math
from Detections import Detections


class Exporter():
    def __init__(self,output_path,filename,detection,classNames, desired_width, desired_height):

        # Initialize variables

        self.output_path=output_path                  #output path + filename
        self.classNames=classNames
        self.desired_width=desired_width
        self.desired_height=desired_height

        self.filename=filename
        # filename = output_path.split('\\')
        # filename = filename[-1].split('.')
        # self.filename = filename[0]

        self.boxes=detection.boxes
        self.classes=detection.classes
        self.confs = detection.confidences

        self.df=self.Output_df()
    ### Output Pascal VOC format for GUI

    def Output_Pascal_VOC(self,merg_img):

        # df = Dataframe storing results of the merged bounding boxes and class names
        # merged_img_path = The path of the merged image. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg'
        # merg_img = The image of interest (In this case: Merged image of the SALTI Program)
        # output_path = Location where pascal voc file needs to be stored. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\output'
        df=self.df
        img_sz = np.shape(merg_img)  # Get the size of the image

        # Writer(path, width, height) -  adds description about the shape of the image
        writer = Writer(self.output_path, img_sz[0], img_sz[1], img_sz[2])

        # For loop to iterate over all the rows of the Dataframe
        for index, row in df.iterrows():
            #::addObject(name, xmin, ymin, xmax, ymax) - Converts information from Dataframe to pascal voc format
            writer.addObject(df.iloc[index, 6], int(df.iloc[index, 2] - df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] - df.iloc[index, 5] / 2), int(df.iloc[index, 2] + df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] + df.iloc[index, 5] / 2))


        # Save the file to output folder
        writer.save(self.output_path + '\\' + self.filename + '.xml')



    ### Output Yolo format for GUI
    def Output_YOLO(self):

        # For GUI
        CL_GUI = ["Category", "xc", "yc", "w", "h"]
        df_GUI = pd.DataFrame(columns=CL_GUI)
        j = 0
        for i in range(len(self.boxes)):
            box, conf, name = self.boxes[i], self.confs[i], self.classes[i]
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[
                3]  # Bounding box is X_topleft,Y_topleft while we need X_cent, Y_cent for GUI
            # For GUI
            df_GUI.at[j, CL_GUI[0]] = name
            df_GUI.at[j, CL_GUI[1]] = x / self.desired_width  # X_centroid for GUI
            df_GUI.at[j, CL_GUI[2]] = y / self.desired_height  # y_centroid for GUI
            df_GUI.at[j, CL_GUI[3]] = w / self.desired_width  # horizontal distance
            df_GUI.at[j, CL_GUI[4]] = h / self.desired_height  # Vertical distance
            j+=1

        return df_GUI

    # In main file do df=Exporter.Output_df() to extract dataframes
    def Output_df(self):

        # Preparing a blank dataframe for each picture's results
        CL = ["Image", "Box", "xc", "yc", "w", "h", "Category", "Confidence"]
        df = pd.DataFrame(columns=CL)
        j = 0
        for i in range(len(self.boxes)):
            box, conf, name = self.boxes[i], self.confs[i], self.classes[i]
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
            df.at[j, CL[6]] = self.classNames[self.classes[i]]
            df.at[j, CL[7]] = self.confs[i]
            j += 1


        return df

    def df_GUI_csv(self):

        df_GUI=self.Output_YOLO()

        # Saving to gui readable format
        df_GUI.to_csv(os.path.join(self.output_path, self.filename + '.txt'), header=None, index=None, sep=' ')
        df_GUI.to_csv(os.path.join(self.output_path, self.filename + '_VAL.txt'), header=None, index=None, sep=' ')

    def df_csv(self):

        df=self.df
        # Exporting each picture's results to its specific csv file
        df.to_csv(os.path.join(self.output_path, self.filename + '.csv'), index=False)


def test_exporter():
    img = cv2.imread(r'C:\Users\20181049\Downloads\Arjun v1.0\Dataset_V0\images\set00\V000\visible\I00000.jpg')
    classnames = ['car']
    det = Detections([[200, 200, 300, 300]], [0], [0.9])
    out_path=r'C:\Users\20181049\Downloads\check'
    filename=r'I00000'
    exp=Exporter(out_path,filename,det,classnames,640,512)
    df=exp.Output_df()
    print(df)
    df_GUI=exp.Output_YOLO()
    print(df_GUI)
    exp.Output_Pascal_VOC(img)
    exp.df_GUI_csv()
    exp.df_csv()


test_exporter()