'''
This file contains all the functions that are used for preprocessing the images
'''
import os
import cv2

#Function to Resize the image
def resize_image(input_image,desired_width,desired_height):
    resized_image = cv2.resize(input_image, (desired_width,desired_height), interpolation=cv2.INTER_AREA)
    return resized_image

def preprocess(image):
    '''
    Insert the preprocessing code
    :param image:
    :return:
    '''
    return image
