from Preprocesser import *

def test_preprocessor():
    output_size = [312, 312]
    img = cv2.imread('D:\KAIST\set00\V000\lwir\I00000.jpg')
    cv2.imshow("TEST", img)
    cv2.waitKey(100)
    PP = Preprocessor(output_size, resize=True)
    img_pp = PP.process(img)
    sz = img_pp.shape
    assert(sz[0]==output_size[0] and sz[1]==output_size[1])
