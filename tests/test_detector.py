from Detector import *

def test_thermal():
    ee = Detector('RGB',0.5,0.5)
    aa=cv2.imread(r'Data\Dataset_V0\images\set00\V000\visible\I00041.jpg')
    bb=YoloJoeHeller(0.1,0.1)
    cc = bb.detect(aa)
    print(cc)
    ee.detect(aa)



def test_rgb():
    ee = Detector('Thermal',0.5,0.5)
    a=cv2.imread(r'Data\Dataset_V0\images\set00\V000\visible\I00041.jpg')
    b=YOLOv3_320()
    c=b.detect(a)
    ee.detect(a)
    print(c)

#test_rgb()
#test_thermal()
