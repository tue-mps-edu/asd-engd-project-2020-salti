from Visualizer import *

def test_visualizer():
    img_c = cv2.imread(r'D:\KAIST\set00\V000\lwir\I00000.jpg')
    img_t = cv2.imread(r'D:\KAIST\set00\V000\visible\I00000.jpg')
    classnames = ['car']
    det = Detections([[200,200,300,300]],[0],[0.9])
    vis = Visualize_all(img_c, img_t)
    vis.print(classnames,det,det,det)

    cv2.waitKey(10)
    print('pause')
