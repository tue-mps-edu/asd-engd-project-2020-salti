# import the necessary packages
from nms import non_max_suppression_slow, non_max_suppression_fast
import numpy as np
import cv2
import time
import xml.etree.ElementTree as ET


## PASCAL VOC coordinates extraction
def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes
name, boxes = read_content("Pascal.xml")
name1,boxes1= read_content("Pascal1.xml")
print(name)
a=np.array(boxes)
b=np.array(boxes1)
bbox=np.append(a,b,axis=0)

 
# construct a list containing the images that will be examined
# along with their respective bounding boxes
images = [
    ("images/Parking.jpg", bbox)]
    # (12, 84, 140, 212),
    # (24, 84, 152, 212),
    # (36, 84, 164, 212),
    # (12, 96, 140, 224),
    # (24, 96, 152, 224),
    # (24, 108, 152, 236),
    # (32, 84, 120, 202),
    # (24, 74, 152, 222),
    # (16, 84, 134, 212),
    # (12, 96, 140, 214),
    # (24, 76, 152, 224),
    # (34, 118, 142, 246)])),
    # ("images/Parking.jpg", np.array([
    #     (361,216,399,243),
    #     (410,234,485,276)]))]

    # (114, 60, 178, 124),
    # (120, 60, 184, 124),
    # (114, 66, 178, 130)]))]
    # ("images/gpripe.jpg", np.array([
    # (12, 30, 76, 94),
    # (12, 36, 76, 100),
    # (72, 36, 200, 164),
    # (84, 48, 212, 176)]))]
    
iter_num= 1
images = images*iter_num  # change the iterations to compare the two nms method

t1 = time.time()
    
# loop over the images
for (i, (imagePath, boundingBoxes)) in enumerate(images):
    # load the image and clone it
    # print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    orig = image.copy()
 
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    
    # perform non-maximum suppression on the bounding boxes
    # pick = non_max_suppression_slow(boundingBoxes, 0.3)
    
    pick = non_max_suppression_fast(boundingBoxes, probs=None, overlapThresh=0.3)

    # print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))   
    
    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
    # display the images
    # cv2.imshow("Original" + i, orig)
    # cv2.imshow("After NMS" + i, image)
    # cv2.waitKey(0)
    
    # save the images
    cv2.imwrite("images/Original_test" + str(i) + ".jpg", orig)
    cv2.imwrite("images/After_NMS_test" + str(i) + ".jpg", image)
    
t2 = time.time()
print('cost {} ms to process {} images'.format((t2 - t1)*1000, len(images)))