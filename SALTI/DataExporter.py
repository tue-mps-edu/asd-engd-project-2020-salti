import cv2
import os
import datetime
import pandas as pd

from pascal_voc_writer import Writer
from shutil import copyfile

class DataExporter():
    def __init__(self, config, output_path, classnames):
        '''
        Data exporter class. This class saves images and labels to the hard drive.

        Arguments:
            (parser) configuration parser object
            (output_path) main folder where subfolders are created for each run
            (list) classnames used for object plotting
        '''
        
        # Initialize variables
        self.make_validation_copy = config['bln_validationcopy']
        self.save_filtered_img = config['bln_savefiltered'] and config['bln_dofilter']
        self.label_type = config['str_label']
        self.classNames = classnames

        #If the output root folder does not exist a new one is created accordingly
        if not os.path.isdir(output_path):
            self.try_to_make_folder(output_path)

        # Create subfolder with date-time stamp
        self.output_subfolder_path = os.path.join(output_path,datetime.datetime.now().strftime('%Y.%m.%d_%Hh%Mm%Ss'))
        self.try_to_make_folder(self.output_subfolder_path)

        # Save configuration for this run into the results folder
        copyfile('config.ini',os.path.join(self.output_subfolder_path,'config.ini'))

        # Create a subfolder for saving filtered images if user wants this
        if self.save_filtered_img:
            self.path_filtered = os.path.join(self.output_subfolder_path,'filtered_images')
            self.try_to_make_folder(self.path_filtered)

        # Generate a file that contains all classnames in the results folder
        df = pd.DataFrame(self.classNames)
        df.to_csv(os.path.join(self.output_subfolder_path,'classes.txt'), header=None, index=None)

    def export(self, output_size, file_name, file_ext, detections, img_raw, img_fil, config):
        '''
        Exporting the YOLO/PascalVOC labels to the results folder

        Arguments:
            (list): output size
            (str): filename
            (str): file extension format
            (Detections): object detections for image
            (opencv img): raw thermal image
            (opencv img): filtered thermal image
        '''
        self.filename = file_name
        self.output_size = output_size
        self.detections = detections

        df_overview = self.create_dataframe_overview()

        if (self.label_type == 'PascalVOC'):
            self.df_label = self.output_pasval_voc(df_overview)

        elif (self.label_type == 'YOLO'):
            self.df_label = self.create_yolo_label()
            self.save_dataframe_as_txt(self.df_label,self.filename)
            if self.make_validation_copy:
                self.save_dataframe_as_txt(self.df_label,self.filename+'_VAL')

        # Save the images
        self.save_opencv_image(self.output_subfolder_path,img_raw,file_ext)
        if self.save_filtered_img:
            self.save_opencv_image(self.path_filtered,img_fil,file_ext)


    def output_pasval_voc(self, df):
        '''
        Write the PascalVOC labels to harddrive

        Arguments:
            (dataframe): PascalVOC label dataframe
        '''
        # Writer(path, width, height) -  adds description about the shape of the image
        writer = Writer(self.output_subfolder_path, self.output_size[0], self.output_size[1], self.output_size[2])

        # For loop to iterate over all the rows of the Dataframe
        for index, row in df.iterrows():
            writer.addObject(df.iloc[index, 6], int(df.iloc[index, 2] - df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] - df.iloc[index, 5] / 2), int(df.iloc[index, 2] + df.iloc[index, 4] / 2),
                             int(df.iloc[index, 3] + df.iloc[index, 5] / 2))

        # Save the file to output folder
        try:
            writer.save(self.output_subfolder_path + '\\' + self.filename + '.xml')
            if self.make_validation_copy:
                writer.save(self.output_subfolder_path + '\\' +  self.filename+'_VAL' + '.xml')
        except:
            raise FileExistsError


    def create_yolo_label(self):
        '''
        Create a dataframe for labels in YOLO format

        Arguments:
            -

        Returns:
            (dataframe) YOLO labels
        '''
        CL_GUI = ["Category", "xc", "yc", "w", "h"]
        df_yolo = pd.DataFrame(columns=CL_GUI)
        j = 0

        # Loop over detections
        for i in range(len(self.detections.boxes)):
            box, conf, name = self.detections.boxes[i], self.detections.confidences[i], self.detections.classes[i]

            # Convert box format from: coordinate_topleft, coordinate_bottomright --> x_center, y_center, width, height
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[3]

            # Convert to YOLO format
            df_yolo.at[j, CL_GUI[0]] = name
            df_yolo.at[j, CL_GUI[1]] = x / self.output_size[0] # X_centroid for GUI
            df_yolo.at[j, CL_GUI[2]] = y / self.output_size[1] # y_centroid for GUI
            df_yolo.at[j, CL_GUI[3]] = w / self.output_size[0] # horizontal distance
            df_yolo.at[j, CL_GUI[4]] = h / self.output_size[1] # Vertical distance
            j+=1

        return df_yolo


    # In main file do df=Exporter.Output_df() to extract dataframes
    def create_dataframe_overview(self):
        '''
        Create a dataframe that logs the bounding boxes for all detections

        Arguments:
            -

        Returns:
            (dataframe): overview of detections
        '''

        # Preparing a blank dataframe for each picture's results
        CL = ["Image", "Box", "xc", "yc", "w", "h", "Category", "Confidence"]
        df_csv = pd.DataFrame(columns=CL)

        # Loop over the detections
        j = 0
        for i in range(len(self.detections.boxes)):
            box, conf, classid = self.detections.boxes[i], self.detections.confidences[i], self.detections.classes[i]

            # Convert box format from: coordinate_topleft, coordinate_bottomright --> x_center, y_center, width, height
            x, y, w, h = int(box[0] + box[2] / 2), int(box[1] + box[3] / 2), box[2], box[3]

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
        '''
        Saving a dataframe to the harddrive as txt format
        Arguments:
            (dataframe) data to be saved
            (string) filename to save as
        '''
        df_txt.to_csv(os.path.join(self.output_subfolder_path, filename + '.txt'), header=None, index=None, sep=' ')

    def save_dataframe_as_csv(self, df_csv, filename):
        '''
        Saving a dataframe to the harddrive as CSV (Comma Separated Values).
        Arguments:
            (dataframe) data to be saved
            (string) filename to save as
        '''
        df_csv.to_csv(os.path.join(self.output_subfolder_path, filename + '.csv'), index=False)

    def save_opencv_image(self, path, img, file_ext):
        '''Write an opencv image to the harddrive'''
        cv2.imwrite(os.path.join(path, self.filename+file_ext), img)

    def try_to_make_folder(self, folder):
        '''Try to make a folder. If this fails an error is printed.'''
        try:
            os.mkdir(folder)
        except:
            print('Failed to create directory: '+folder)
