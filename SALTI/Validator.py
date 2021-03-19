import numpy as np
import os
import pandas as pd

class Validator():
    def __init__(self,Results_directory, img_extention,iou_threshold):
        self.Results_directory=Results_directory
        self.img_extention=img_extention
        self.iou_threshold=iou_threshold

    def get_iou(self,pred_box, gt_box):
        """
        pred_box : the coordinate for predict bounding box
        gt_box :   the coordinate for ground truth bounding box
        return :   the iou score
        the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
        the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
        """
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


    def single_Validate(self,file):
        #Initializing the TP, FP and FN values
        TP = 0
        FP = 0
        FN = 0

        file_name = os.path.splitext(file)[0]
        # GUI
        GU = pd.read_csv(os.path.join(self.Results_directory, file_name + ".txt"), delim_whitespace=True,
                         names=["classes", "x_c", "y_c", "w", "h"])
        GU['filename'] = file_name

        # Yolo
        YO = pd.read_csv(os.path.join(self.Results_directory, file_name + "_VAL.txt"), delim_whitespace=True,
                         names=["classes", "x_c", "y_c", "w", "h"])
        YO['filename'] = file_name


        distances_mat = np.zeros((GU.shape[0], YO.shape[0]))

        for index_GU in range(GU.shape[0]):  # 0 to 2 (along columns of gui)
            for index_YO in range(YO.shape[0]):  # 0 to 3 (along columns of yolo)
                centroid_distances = np.sqrt((GU['x_c'][index_GU] - YO['x_c'][index_YO]) ** 2 + (
                        GU['y_c'][index_GU] - YO['y_c'][index_YO]) ** 2)
                distances_mat[index_GU][index_YO] = centroid_distances

        # Creating a dataframe for distances between GUI (in rows) boxes and Yolo (in column) boxes
        distances_df = pd.DataFrame(data=distances_mat)

        while distances_df.shape[0] >= 1 and distances_df.shape[1] >= 1:

            # To find the indexes of global minimum inside the dataframe
            GUI_min_ind = distances_df.min(axis=1).idxmin()
            YO_min_ind = distances_df.min().idxmin()

            # calculate the edges of the YOLO and GUI boxes
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
            '''
            #compare the minimum with diameter of the larger box
            GUI_diam = np.sqrt( (GU['w'][GUI_min_ind]) **2 + (GU['h'][GUI_min_ind]) **2 )
            YO_diam = np.sqrt((YO['w'][YO_min_ind]) ** 2 + (YO['h'][YO_min_ind]) ** 2)
            bigger_diameter = max(GUI_diam, YO_diam)
            #if distances_df[GUI_min_ind][YO_min_ind] < bigger_diameter:
            '''
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

        # metrics calculations
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

        # print('Image ' + file_name + ': TP = {}, FP is {}, FN is {}'.format(TP, FP, FN))
        # print('Image ' + file_name + ': Precision is {}, Recall is {}, Accuracy is {} and F1 is {}'.format(Precision,
        #                                                                                                   Recall,
        #                                                                                                   Accuracy,
        #                                                                                                   F1_score))
        return TP,FP,FN

    def complete_Validation(self):
        # this is code for reading output files from YOlO and from the GUI
        TP_tot = 0
        FP_tot = 0
        FN_tot = 0

        for file in os.listdir(self.Results_directory):  # loop through directory of images
            if os.path.splitext(file)[1] not in self.img_extention:
                continue
            print('Validating the image: ' + file)

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
        return Precision_tot, Recall_tot

Precision_ROC = np.array([])
Recall_ROC = np.array([])
validation_threshold = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
j = 0
for i in validation_threshold:
    v = Validator(r'D:\Courses\Block 2\Inhouse-project\GIT\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_DAY\Output\2021.03.19_11h18m32s','.jpg', i)
    precision_roc, recall_roc = v.complete_Validation()
    Precision_ROC = np.append(Precision_ROC,precision_roc)
    Recall_ROC = np.append(Recall_ROC,recall_roc)
    j = j + 1

print(Precision_ROC, Recall_ROC)



