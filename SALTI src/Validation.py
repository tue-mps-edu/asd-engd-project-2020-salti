import numpy as np
from config import *
#from localFunctions import *
import pandas as pd
import os

# from sklearn import metrics
import shutil


def get_iou(pred_box, gt_box):
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

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def UserValidation(Results_directory,img_extention):
    #this is code for reading output files from YOlO and from the GUI

    TP_tot = 0
    FP_tot = 0
    FN_tot = 0

    for file in os.listdir(Results_directory): #loop through directory of images
        if os.path.splitext(file)[1] not in img_extention:
            continue
        print(file)
        file_name = os.path.splitext(file)[0]

        # GUI
        GU = pd.read_csv(os.path.join(Results_directory,file_name + ".txt"), delim_whitespace=True,
            names=["classes", "x_c", "y_c", "w", "h"])
        print(GU)
        GU['filename'] = file_name


        #Yolo
        YO = pd.read_csv(os.path.join(Results_directory,file_name + "_VAL.txt"), delim_whitespace=True,
                    names=["classes", "x_c", "y_c", "w", "h"])
        YO['filename'] = file_name

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

        while distances_df.shape[0] >= 1 and distances_df.shape[1] >= 1:

            # To find the indexes of global minimum inside the dataframe
            GUI_min_ind = distances_df.min(axis=1).idxmin()
            YO_min_ind = distances_df.min().idxmin()

            #calculate the edges of the YOLO and GUI boxes
            x_GU_low = GU['x_c'][GUI_min_ind] - (GU['w'][GUI_min_ind])*1/2
            y_GU_low = GU['y_c'][GUI_min_ind] - (GU['h'][GUI_min_ind]) * 1 / 2
            x_GU_high = GU['x_c'][GUI_min_ind] + (GU['w'][GUI_min_ind]) * 1 / 2
            y_GU_high = GU['y_c'][GUI_min_ind] + (GU['h'][GUI_min_ind]) * 1 / 2

            x_YO_low = YO['x_c'][YO_min_ind] - (YO['w'][YO_min_ind])*1/2
            y_YO_low = YO['y_c'][YO_min_ind] - (YO['h'][YO_min_ind]) * 1 / 2
            x_YO_high = YO['x_c'][YO_min_ind] + (YO['w'][YO_min_ind]) * 1 / 2
            y_YO_high = YO['y_c'][YO_min_ind] + (YO['h'][YO_min_ind]) * 1 / 2

            #Calculate the IoU (predicted YO box vs ground truth GU box) and set threshold
            IoU = get_iou([x_YO_low,y_YO_low,x_YO_high,y_YO_high], [x_GU_low, y_GU_low, x_GU_high, y_GU_high])
            print('The IoU is {}'.format(IoU))
            threshold = 0.8
            '''
            #compare the minimum with diameter of the larger box
            GUI_diam = np.sqrt( (GU['w'][GUI_min_ind]) **2 + (GU['h'][GUI_min_ind]) **2 )
            YO_diam = np.sqrt((YO['w'][YO_min_ind]) ** 2 + (YO['h'][YO_min_ind]) ** 2)
            bigger_diameter = max(GUI_diam, YO_diam)
            #if distances_df[GUI_min_ind][YO_min_ind] < bigger_diameter:
            '''
            if IoU > threshold:
                #They are the same boxes

                if GU['classes'][GUI_min_ind]==YO['classes'][YO_min_ind]:
                    TP+=1
                    TP_tot += 1

                else:
                    FP+=1
                    FP_tot+=1

                # To discard the analyzed row and column
                distances_df = distances_df.drop(index=[GUI_min_ind], columns=[YO_min_ind])
            else:
                FN+=1
                FP+=1

                FN_tot+=1
                FP_tot+=1

                distances_df = distances_df.drop(index=[GUI_min_ind], columns=[YO_min_ind])

        FN+=distances_df.shape[0]
        FP+=distances_df.shape[1]

        FN_tot+=distances_df.shape[0]
        FP_tot+=distances_df.shape[1]

        #metrics calculations
        if FP == 0:
            Precision = 1
        else:
            Precision = TP / (TP + FP)
        if FN == 0:
            Recall = 1
        else:
            Recall = TP / (TP + FN)
        if FN+TP+FP == 0:
            Accuracy = 1
        else:
            Accuracy = TP / (TP + FP + FN)

        F1_score = (2*(Precision*Recall))/(Precision+Recall)
        print('Image'+file_name+': TP = {}, FP is {}, FN is {}'.format(TP, FP, FN))
        print('Image'+file_name+': Precision is {}, Recall is {}, Accuracy is {} and F1 is {}'.format(Precision, Recall, Accuracy, F1_score))

    #Overall metrics calculations
    if FP_tot == 0:
        Precision_tot = 1
    else:
        Precision_tot = TP_tot/ (TP_tot+ FP_tot)
    if FN_tot == 0:
        Recall_tot = 1
    else:
        Recall_tot = TP_tot / (TP_tot + FN_tot)
    if FN_tot + TP_tot + FP_tot == 0:
        Accuracy_tot = 1
    else:
        Accuracy_tot = TP_tot / (TP_tot + FP_tot + FN_tot)
    F1_score_tot = (2*(Precision_tot*Recall_tot))/(Precision_tot+Recall_tot)
    print('Total: TP = {}, FP is {}, FN is {}'.format(TP_tot, FP_tot, FN_tot))
    print('Total: Precision is {}, Recall is {}, Accuracy is {} and F1 is {}'.format(Precision_tot, Recall_tot, Accuracy_tot, F1_score_tot))
    # mAP = metrics.auc(Precision_tot, Recall_tot)
    # print(mAP)

img_extention = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
UserValidation(dir_Validation,img_extention)


#read_and_display_boxes(dir_Validation, 'I02199')

