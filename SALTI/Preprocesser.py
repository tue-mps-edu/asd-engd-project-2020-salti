import cv2
import numpy as np
import os

class Preprocessor():

    def __init__(self,output_size=[640,512],  enhancing=False):
        self.output_size = output_size  # [x,y]
        self.do_enhancing = enhancing

    def process(self, img):
        self.do_resize = not(img.shape[0] == self.output_size[0] and img.shape[1] == self.output_size[1])
        if self.do_resize:
            img = self.resize_image(img)
        if self.do_enhancing:
            img_enh = self.add_enhancing(img)
            return img, img_enh
        else:
            return img, img

    def resize_image(self, img_in):
        return cv2.resize(img_in, (self.output_size[0],self.output_size[1]), interpolation=cv2.INTER_AREA)

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
        cv2.imwrite(os.path.join(r'D:\PDEng2020-2022\Block 2\In-House project\ASD SCRUM\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_NIGHT\saved_pp','saved.jpg'), img_shr_rgbeq)
        return img_shr_rgbeq



