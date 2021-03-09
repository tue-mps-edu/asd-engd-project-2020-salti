import cv2

class Preprocessor():
    def __init__(self,output_size=[640,512],resize=False, padding=False):
        self.output_size = output_size  # [x,y]
        self.do_resize = resize
        self.do_padding = padding

    def process(self, img):
        if self.do_resize:
            img = self.resize_image(img)
        if self.do_padding:
            img = self.add_padding(img)
        return img

    def resize_image(self, img_in):
        return cv2.resize(img_in, (self.output_size[0],self.output_size[1]), interpolation=cv2.INTER_AREA)

    def add_padding(self, img_in):
        return img_in

def test_preprocessor():
    output_size = [312, 312]
    img = cv2.imread('D:\KAIST\set00\V000\lwir\I00000.jpg')
    cv2.imshow("TEST", img)
    cv2.waitKey(100)
    PP = Preprocessor(output_size, resize=True)
    img_pp = PP.process(img)
    sz = img_pp.shape
    assert(sz[0]==output_size[0] and sz[1]==output_size[1])
