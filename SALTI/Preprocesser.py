import cv2

class Preprocessor():
    def __init__(self,output_size=[640,512]):
        self.output_size = output_size  # [x,y]

    def process(self, img, do_resize=False, do_filter=False):
        if do_resize:
            img = self.resize_image(img)
        if do_filter:
            img = self.filter_image(img)
        return img

    def resize_image(self, img_in):
        return cv2.resize(img_in, (self.output_size[0],self.output_size[1]), interpolation=cv2.INTER_AREA)

    def filter_image(self, img_in):
        return img_in
