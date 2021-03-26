import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as ET
import sys


class Validator():
    def __init__(self,Results_directory, img_extention,iou_threshold,label_type):
        '''
        Validator class that compares a set of labels to see what differences there are in the boxes.
        Using the IOU value of the boxes, statistics are calculated:
        Accuracy, Precision, F1, Recall

        For the validator the work, a folder needs to contain:
        1. the images (e.g. I00000.jpg)
        2. a label file that is corrected by human (e.g. I00000.txt / I00000.xml)
        3. a validation label file provided optionally by SALTI (e.g. I00000_val.txt)

        argument 1: directory to labels and images
        argument 2: image extension (e.g. .jpg)
        argument 3: IOU threshold (default=0.9)
        argument 4: label type, Yolo/PascalVOC
        '''

        self.Results_directory=Results_directory
        self.img_extention=img_extention
        self.iou_threshold=iou_threshold
        self.classNames = self.read_classes_txt(os.path.join(self.Results_directory, 'classes.txt'))
        self.label_type = label_type

    def get_iou(self,pred_box, gt_box):
        '''
        Reference: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py

        This function takes asn an input the predicted bounding box and the ground truth bounding box
        and evaluates the Intersection over Union (IoU) between the two:

        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        '''
        # 1.get the coordinate of inters
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # 2. calculate the area of inters
        inters = iw * ih

        # 3. calculate the area of union
        uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

        # 4. calculate the overlaps between pred_box and gt_box
        iou = inters / uni
        return iou

    def read_classes_txt(self,classes_txt_file):
        '''Read the class names from classes.txt file in the results folder'''
        with open(classes_txt_file) as classes:
            classNames = classes.readlines()

        # To remove whitespace characters like `\n` at the end of each line
        classNames = [x.strip() for x in classNames]
        return classNames

    def read_xml_label(self,xml_file):
        '''Read information from the xml file (PASCALVOC label format) and store the information in a dataframe'''

        tree = ET.parse(xml_file)
        root = tree.getroot()

        tot_width = int(root.find("size/width").text)
        tot_height = int(root.find("size/height").text)

        CL = ["classes", "x_c", "y_c", "w", "h"]
        df = pd.DataFrame(columns=CL)

        for boxes in root.iter('object'):

            object = boxes.find("name").text
            object_index = self.classNames.index(object)

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)

            x_c = (xmin + xmax) / 2 / tot_width
            y_c = (ymin + ymax) / 2 / tot_height
            box_width = (xmax - xmin) / tot_width
            box_height = (ymax - ymin) / tot_height

            df = df.append(pd.Series(
                [object_index, x_c, y_c, box_width, box_height],
                index=df.columns), ignore_index=True)

        return df

    def read_txt_label(self,txt_file):
        '''Read information from the text file (YOLO label format) and store the information in a dataframe'''

        df = pd.read_csv(txt_file, delim_whitespace=True,
                         names=["classes", "x_c", "y_c", "w", "h"])
        return df


    def single_Validate(self,img_file):
        ''' Function to perform validation for a single image '''

        #Initializing the TP, FP and FN values
        TP = 0
        FP = 0
        FN = 0

        file_name = os.path.splitext(img_file)[0]   # Obtain the filename of the current image to be validated

        if self.label_type.lower() == 'yolo':   # Check for label type
            GU = self.read_txt_label(os.path.join(self.Results_directory, file_name + ".txt"))  # Dataframe corresponding to user-modiphied GUI predictions
            YO = self.read_txt_label(os.path.join(self.Results_directory, file_name + "_VAL.txt")) # Dataframe corresponding to initial Yolo predictions
        elif self.label_type.lower() == 'pascalvoc': # Check for label type
            GU = self.read_xml_label(os.path.join(self.Results_directory, file_name + ".xml")) # Dataframe corresponding to user-modiphied GUI predictions
            YO = self.read_xml_label(os.path.join(self.Results_directory, file_name + "_VAL.xml")) # Dataframe corresponding to initial Yolo predictions
        else:
            print("Label type not included")

        distances_mat = np.zeros((GU.shape[0], YO.shape[0]))      # Initialize the distance matrix

        for index_GU in range(GU.shape[0]):  # Along columns of gui
            for index_YO in range(YO.shape[0]):  # Along columns of yolo
                centroid_distances = np.sqrt((GU['x_c'][index_GU] - YO['x_c'][index_YO]) ** 2 + (GU['y_c'][index_GU] - YO['y_c'][index_YO]) ** 2) # Compute the centroid distances
                distances_mat[index_GU][index_YO] = centroid_distances  # Assign the centroid distance information in the distance matrix

        # Creating a dataframe for distances between GUI (in rows) boxes and Yolo (in column) boxes
        distances_df = pd.DataFrame(data=distances_mat)

        while distances_df.shape[0] >= 1 and distances_df.shape[1] >= 1:

            # To find the indexes of global minimum inside the dataframe
            GUI_min_ind = distances_df.min(axis=1).idxmin()
            YO_min_ind = distances_df.min().idxmin()

            # Calculate the edges of the YOLO and GUI boxes
            x_GU_low = GU['x_c'][GUI_min_ind] - (GU['w'][GUI_min_ind]) * 1 / 2
            y_GU_low = GU['y_c'][GUI_min_ind] - (GU['h'][GUI_min_ind]) * 1 / 2
            x_GU_high = GU['x_c'][GUI_min_ind] + (GU['w'][GUI_min_ind]) * 1 / 2
            y_GU_high = GU['y_c'][GUI_min_ind] + (GU['h'][GUI_min_ind]) * 1 / 2

            x_YO_low = YO['x_c'][YO_min_ind] - (YO['w'][YO_min_ind]) * 1 / 2
            y_YO_low = YO['y_c'][YO_min_ind] - (YO['h'][YO_min_ind]) * 1 / 2
            x_YO_high = YO['x_c'][YO_min_ind] + (YO['w'][YO_min_ind]) * 1 / 2
            y_YO_high = YO['y_c'][YO_min_ind] + (YO['h'][YO_min_ind]) * 1 / 2

            # Calculate the IoU (predicted YO box vs ground truth GU box) and set threshold
            IoU = self.get_iou([x_YO_low, y_YO_low, x_YO_high, y_YO_high], [x_GU_low, y_GU_low, x_GU_high, y_GU_high])

            if IoU > self.iou_threshold:
                # They are the same boxes

                if GU['classes'][GUI_min_ind] == YO['classes'][YO_min_ind]:
                    TP += 1
                else:
                    FP += 1

                # To discard the analyzed row and column
                distances_df = distances_df.drop(index=[GUI_min_ind], columns=[YO_min_ind])
            else:
                FN += 1
                FP += 1

                distances_df = distances_df.drop(index=[GUI_min_ind], columns=[YO_min_ind])

        FN += distances_df.shape[0]
        FP += distances_df.shape[1]

        # Metrics calculations
        if TP + FP == 0:
            Precision = 1
        else:
            Precision = TP / (TP + FP)

        if TP + FN == 0:
            Recall = 1
        else:
            Recall = TP / (TP + FN)

        if TP + FP + FN == 0:
            Accuracy = 1
        else:
            Accuracy = TP / (TP + FP + FN)

        if Precision + Recall == 0:
            F1_score=0
        else:
            F1_score = (2 * (Precision * Recall)) / (Precision + Recall)

        print('Image ' + file_name + ': TP = {}, FP is {}, FN is {}'.format(TP, FP, FN))
        print('Image ' + file_name + ': Precision is {}, Recall is {}, Accuracy is {} and F1 is {}'.format(Precision,
                                                                                                          Recall,
                                                                                                          Accuracy,
                                                                                                          F1_score))
        return TP,FP,FN     # Return the TP,FP,FN information for the current image

    def complete_Validation(self):
        '''Function to perform single validation script for each image in the directory'''
        TP_tot = 0
        FP_tot = 0
        FN_tot = 0

        for file in os.listdir(self.Results_directory):  # Loop through files in the validation directory
            if os.path.splitext(file)[1] != self.img_extention:
                continue
            print('Validating the image: ' + file)

            # Run the single validate function for the current file
            TP,FP,FN = self.single_Validate(file)

            TP_tot += TP
            FP_tot += FP
            FN_tot += FN

        # Overall metrics calculations
        if TP_tot + FP_tot == 0:
            Precision_tot = 1
        else:
            Precision_tot = TP_tot / (TP_tot + FP_tot)

        if TP_tot + FN_tot == 0:
            Recall_tot = 1
        else:
            Recall_tot = TP_tot / (TP_tot + FN_tot)

        if TP_tot + FP_tot + FN_tot == 0:
            Accuracy_tot = 1
        else:
            Accuracy_tot = TP_tot / (TP_tot + FP_tot + FN_tot)

        if Precision_tot + Recall_tot == 0:
            F1_score_tot = 0
        else:
            F1_score_tot = (2 * (Precision_tot * Recall_tot)) / (Precision_tot + Recall_tot)

        print('Total: TP = {}, FP is {}, FN is {}'.format(TP_tot, FP_tot, FN_tot))
        print('Total: Precision is {}, Recall is {}, Accuracy is {} and F1 is {}'.format(Precision_tot, Recall_tot,
                                                                                         Accuracy_tot, F1_score_tot))



def Validate(directory,img_ext,IOU_threshold,label_type):
    """
        Function to validate the whole given directory
            directory : the directory that contains the files needed to be validated
            img_ext :   the image extension (for example : '.jpg')
            IOU_threshold :   the iou threshold that determines how much overlap is to be considered as a true positive (TP)
            label_type : the type of labels to be validated (for example : 'YOLO' or 'PascalVOC'
    """
    v=Validator(directory,img_ext,IOU_threshold,label_type)
    v.complete_Validation()


'''
If you want to run the validator directly from your IDE then uncomment the block below and put as input
the directory, image extension and IOU_threshold.
Remember to comment the windows command line section as well. 
'''
# #Initializing the validator
# v = Validate(r'C:\Github\asd-pdeng-project-2020-developer\test_images\outputs\2021.03.26_15h36m33s',
#               '.jpg',
#               0.9,
#              'PasCalVoc')


'''
If you want to use the windows command line using arguments you need to:
1. Open Anaconda Prompt
2. Activate the proper interpreter for example:
            conda activate thermal-joe
3. set the current directory to the folder you have the script in with:
            cd your_directory
4. run the Validator script along with the input arguments as the following:
            python Validator.py "directory" "image_extension" IOU_threshold "label format (Yolo or PascalVOC)"
'''
#Windows command line using arguments
if __name__ == "__main__":
    directory = sys.argv[1]
    img_ext = sys.argv[2]
    IOU_threshold= float(sys.argv[3])
    label_type = sys.argv[4]
    Validate(directory, img_ext, IOU_threshold,label_type)


