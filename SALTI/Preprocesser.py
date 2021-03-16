import cv2

class Preprocessor():
    def __init__(self,output_size=[640,512],resize=False padding=False):
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
