import pandas as pd
from config_thermal.utils_thermal.utils import *
from pascal_voc_writer import Writer
import cv2
import numpy as np
from Detections import Detections
import pandas as pd
import datetime
from shutil import copyfile
import os

class DataExporter():
    def __init__(self,label_type, output_path, classnames):
        # Initialize variables
        self.label_type = label_type
        self.output_path = os.path.join(output_path,datetime.datetime.now().strftime('%Y.%m.%d_%Hh%Mm%Ss'))
        self.classNames = classnames
        self.try_to_make_folder(self.output_path)

        copyfile('config.ini',os.path.join(self.output_path,'config.ini'))

        if (self.label_type == 'YOLO'):
            df = pd.DataFrame(self.classNames)
            df.to_csv(os.path.join(self.output_path,'classes.txt'), header=None, index=None)

    def export(self, output_size, file_name, file_ext, detections, img):
        self.filename = file_name
        self.output_size = output_size
        self.detections = detections

        df_overview = self.create_dataframe_overview()

        if (self.label_type == 'PascalVOC'):
            self.df_label = self.output_pasval_voc(df_overview)

        elif (self.label_type == 'YOLO'):
            self.df_label = self.create_yolo_label()
            self.save_dataframe_as_txt(self.df_label,self.filename)
            self.save_dataframe_as_txt(self.df_label,self.filename+'_VAL')

        # Save the image
        self.save_opencv_image(img,file_ext)

    def output_pasval_voc(self, df):
    ### Output Pascal VOC format for GUI
        # df = Dataframe storing results of the merged bounding boxes and class names
        # merged_img_path = The path of the merged image. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\thermal\I00000.jpg'
        # merg_img = The image of interest (In this case: Merged image of the SALTI Program)
        # output_path = Location where pascal voc file needs to be stored. Ex: 'D:\Git repos\asd-pdeng-project-2020-developer\SALTI src\output'
        #img_sz = np.shape(merg_img)  # Get the size of the image

        # Writer(path, width, height) -  adds description about the shape of the image
        writer = Writer(self.output_path, self.output_size[0], self.output_size[1], self.output_size[2])

        # For loop to iterate over all the rows of the Dataframe
        for index, row in df.iterrows():
            #::addObject(name, xmin, ymin, xmax, ymax) - Converts information from Dataframe to pascal voc format
            writer.addObject(df.iloc[index, 6], int(df.iloc[index, 2] - df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] - df.iloc[index, 5] / 2), int(df.iloc[index, 2] + df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] + df.iloc[index, 5] / 2))

        # Save the file to output folder
        try:
            writer.save(self.output_path + '\\' + self.filename + '.xml')
            writer.save(self.output_path + '\\' +  self.filename+'_val' + '.xml')
        except:
            print('stop here ')

    def create_yolo_label(self):
    ### Output Yolo format for GUI
        # For GUI
        CL_GUI = ["Category", "xc", "yc", "w", "h"]
        df_yolo = pd.DataFrame(columns=CL_GUI)
        j = 0
        for i in range(len(self.detections.boxes)):
            box, conf, name = self.detections.boxes[i], self.detections.confidences[i], self.detections.classes[i]
            # Bounding box is X_topleft,Y_topleft while we need X_cent, Y_cent for GUI
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[3]
            # For GUI
            df_yolo.at[j, CL_GUI[0]] = name
            df_yolo.at[j, CL_GUI[1]] = x / self.output_size[0]#output_output_width  # X_centroid for GUI
            df_yolo.at[j, CL_GUI[2]] = y / self.output_size[1]#output_height  # y_centroid for GUI
            df_yolo.at[j, CL_GUI[3]] = w / self.output_size[0]#output_width  # horizontal distance
            df_yolo.at[j, CL_GUI[4]] = h / self.output_size[1]#output_height  # Vertical distance
            j+=1

        return df_yolo

    # In main file do df=Exporter.Output_df() to extract dataframes
    def create_dataframe_overview(self):

        # Preparing a blank dataframe for each picture's results
        CL = ["Image", "Box", "xc", "yc", "w", "h", "Category", "Confidence"]
        df_csv = pd.DataFrame(columns=CL)
        j = 0
        for i in range(len(self.detections.boxes)):
            box, conf, classid = self.detections.boxes[i], self.detections.confidences[i], self.detections.classes[i]
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[
                3]  # Bounding box is X_topleft,Y_topleft while we need X_cent, Y_cent for GUI

            # Storing each picture's results in its dataframe
            df_csv = df_csv.append(pd.Series(0, index=df_csv.columns), ignore_index=True)
            df_csv.at[j, CL[0]] = self.filename
            df_csv.at[j, CL[1]] = j + 1
            df_csv.at[j, CL[2]] = x  # X_centroid for GUI
            df_csv.at[j, CL[3]] = y  # y_centroid for GUI
            df_csv.at[j, CL[4]] = w  # horizontal distance
            df_csv.at[j, CL[5]] = h  # Vertical distance
            df_csv.at[j, CL[6]] = self.classNames[classid]
            df_csv.at[j, CL[7]] = conf
            j += 1

        return df_csv

    def save_dataframe_as_txt(self, df_txt, filename):
        # Saving to gui readable format
        df_txt.to_csv(os.path.join(self.output_path, filename + '.txt'), header=None, index=None, sep=' ')

    def save_dataframe_as_csv(self, df_csv, filename):
        # Exporting each picture's results to its specific csv file
        df_csv.to_csv(os.path.join(self.output_path, filename + '.csv'), index=False)

    def save_opencv_image(self, img, file_ext):
        cv2.imwrite(os.path.join(self.output_path,self.filename+file_ext), img)

    def try_to_make_folder(self, folder):
        try:
            os.mkdir(self.output_path)
        except:
            assert("Folder already exists")
