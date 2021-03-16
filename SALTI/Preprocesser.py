import cv2
import numpy as np

class Preprocessor():

    def __init__(self,output_size=[640,512], resize=False, padding=False, enhancing=False):
        self.output_size = output_size  # [x,y]
        self.do_resize = resize
        self.do_padding = padding
        self.do_enhancing = enhancing

    def process(self, img, output_size):

        if not(img.shape[0] == output_size[0] and img.shape[1] == output_size[1]):
            if self.do_resize:
                img = self.resize_image(img)

            if self.do_padding:
                img = self.add_padding(img)

        if self.do_enhancing:
            img = self.add_enhancing(img)

        return img

    def resize_image(self, img_in):
        return cv2.resize(img_in, (self.output_size[0],self.output_size[1]), interpolation=cv2.INTER_AREA)

    def filter_image(self, img_in):
        return img_in

    def add_enhancing(self, img_in):

        # Determine average intensity level of the image
        def find_energy(img_gray):
            img_gray = img_gray.astype('float64')
            energy = np.sqrt(np.mean(img_gray ** 2))
            return energy

        # Sharpen based on the normalized image intensity level
        def adaptive_sharpen(img_gray_received):
            img_sharpened = img_gray_received.copy() * 1.0
            blurred = cv2.GaussianBlur(img_sharpened, (5, 5), 2) * 1.0
            img_sharpened = 2.0 * img_sharpened - 1.6 * blurred
            sharp_energy = find_energy(img_sharpened)
            sharp_factor = 100 / sharp_energy
            img_sharpened = img_sharpened * sharp_factor
            img_sharpened[img_sharpened <= 0] = 0
            img_sharpened[img_sharpened >= 255] = 255
            img_sharpened = img_sharpened.astype(np.uint8)
            return img_sharpened

        img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.uint8)

        img_gray_sharpened = adaptive_sharpen(img_gray)

        # The sharpened gray scale image is converted to have 3 channels to prevent dimension mismatch
        img_shr_rgbeq = cv2.cvtColor(img_gray_sharpened, cv2.COLOR_GRAY2RGB)

        return img_shr_rgbeq



