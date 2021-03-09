import cv2
from Detections import Detections

class Visualizer():
    def __init__(self,img):
        self.img=img  # Common image for drawing boxes and printing

    def draw_bboxs(self, classNames, detection):
        for i in range(len(detection.boxes)):
            box, conf, name = detection.boxes[i], detection.confidences[i], detection.classes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 255),
                          2)  # rectangle(starting point which is top left, ending point which is bottom right, color , thickness)
            cv2.putText(self.img, f'{classNames[detection.classes[i]].upper()} {int(detection.confidences[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    def print_annotated_image(self, img_type, classnames, detection):
        self.draw_bboxs(classnames,detection)
        cv2.imshow(img_type,self.img)

class Visualize_all():
    def __init__(self,img_c, img_t):
        self.color = Visualizer(img_c)
        self.thermal = Visualizer(img_t)

    def print(self, classnames, det_c, det_t, det_m):
        self.color.print_annotated_image('RGB', classnames, det_c)
        self.thermal.print_annotated_image('Thermal', classnames, det_t)
        self.thermal.print_annotated_image('Merged', classnames, det_m)
        cv2.waitKey(1000)

def test_visualizer():
    img_c = cv2.imread(r'D:\KAIST\set00\V000\lwir\I00000.jpg')
    img_t = cv2.imread(r'D:\KAIST\set00\V000\visible\I00000.jpg')
    classnames = ['car']
    det = Detections([[200,200,300,300]],[0],[0.9])
    vis = Visualize_all(img_c, img_t)
    vis.print(classnames,det,det,det)

    cv2.waitKey(1000)
    print('pause')


#test_visualizer()
