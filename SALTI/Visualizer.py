import cv2
import numpy as np
from Detections import Detections

class ProgressWindow():
    def __init__(self,img_c, img_t, det_c, det_t, det_m, classnames, progress):
        self.imgs = [img_c.copy(), img_t.copy(), img_t.copy()]
        self.dets = [det_c, det_t, det_m]
        self.titles = ['Color', 'Thermal', 'Thermal merged labels']
        # Add BBOXes to the images
        for img, det, title in zip(self.imgs, self.dets, self.titles):
            self.add_bboxes_to_image(classnames, img, det)
            cv2.putText(img=img, text=title, org=(25,50),fontFace=1, fontScale=2, color=(255,255,255), thickness=2)

        # Shrink the images
        self.imgs_rz = []
        image_scaling = 0.7
        for img in self.imgs:
            self.imgs_rz.append(cv2.resize(img, (0,0), None, image_scaling, image_scaling))

        # Append and draw window
        self.create_window(progress, classnames)

    def add_bboxes_to_image(self,classNames, img, detection):
        for i in range(len(detection.boxes)):
            box, conf, name = detection.boxes[i], detection.confidences[i], detection.classes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255),
                          2)  # rectangle(starting point which is top left, ending point which is bottom right, color , thickness)
            cv2.putText(img, f'{classNames[detection.classes[i]].upper()} {int(detection.confidences[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    def create_window(self, progress, classnames):
        stacked_imgs = np.hstack((self.imgs_rz[0], self.imgs_rz[1], self.imgs_rz[2]))
        self.create_bar_from_img(stacked_imgs, progress)
        self.window = np.concatenate((stacked_imgs, self.bar_progress),axis=0)
        cv2.imshow('merged', self.window)
        cv2.waitKey(1)

    def create_bar_from_img(self, stacked_imgs, progress):
        # Get size of the stacked images
        img_size = stacked_imgs.shape
        # Define size of progress bar
        width_total = img_size[1]
        height = 50
        w_done = round(progress*width_total)
        w_todo = width_total-w_done
        # Create 3D image arrays
        arr_done = 100*np.ones((height,w_done ,3),dtype='uint8')
        arr_done[:,:,0] = 0
        arr_done[:,:,2] = 0
        arr_todo = 50*np.ones((height,w_todo,3),dtype='uint8')
        # Stack done and todo
        bar_progress = np.hstack((arr_done, arr_todo))
        # Add progress text in bar
        cv2.putText(img=bar_progress, text=str(round(progress*100))+'%', org=((round(width_total/2)-20),35),fontFace=1,fontScale=2, color=(255,255,255), thickness=2)
        self.bar_progress = bar_progress


