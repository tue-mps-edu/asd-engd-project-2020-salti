import os
import cv2
import glob

input_formats = ['.png','.jpg','.jpeg']


class Dataloader():
    def __init__(self, path_rgb, path_thermal, output_size=[-1,-1], resize=False, debug=False):
        self.DEBUG = debug                      # Debug flag for skipping images
        self.count = 0                          # Logging the current image
        self.progress = 0                       # for progress bar
        self.path_rgb = path_rgb                # path to rgb directory
        self.path_thermal = path_thermal        # path to thermal directory
        self.resize_images = resize             # BOOLEAN whether images should be resized
        self.image_output_size = output_size    # Output size of the image

        # Only directories are allowed as input
        assert(os.path.isdir(path_rgb) and os.path.isdir(path_thermal))

        # Get the file list
        files_rgb = sorted(glob.glob(os.path.join(path_rgb,'*.*')))
        files_thermal = sorted(glob.glob(os.path.join(path_thermal, '*.*')))

        # Get a list of the images
        self.imgs_thermal = [x for x in files_thermal if os.path.splitext(x)[-1].lower() in input_formats]
        self.imgs_rgb = [x for x in files_rgb if os.path.splitext(x)[-1].lower() in input_formats]
        # Check if there is an equal number of images for rgb-thermal
        assert(len(self.imgs_thermal)==len(self.imgs_rgb))
        self.nr_imgs = len(self.imgs_rgb)

    def __iter__(self):
    # ITERATOR function to use it in a for loop
    # for img_c, img_t in dataloader:
        for i in range(self.nr_imgs):
            self.count = i
            if self.DEBUG:
                if i%60!=0:
                    continue

            path_C = self.imgs_rgb[self.count]              # Color image path
            path_T = self.imgs_thermal[self.count]          # Thermal image path
            if self.DEBUG:
                print('color image: \t'+path_C)
                print('thermal iamge: \t'+path_T)

            img_C = cv2.imread(path_C)                      # Read color image
            img_T = cv2.imread(path_T)                      # Read thermal image

            if self.resize_images:
                img_C = resize_image(img_C, self.image_output_size)
                img_T = resize_image(img_T, self.image_output_size)

            self.count = self.count + 1                     # Update to next image
            self.progress = (self.count/self.nr_imgs)*100   # Update progress
            yield img_C, img_T

def test_dataloader():
    path_t = r'D:\KAIST\set00\V000\lwir'
    path_c = r'D:\KAIST\set00\V000\visible'
    data = Dataloader(path_c, path_t, output_size=[320,320], resize=True, debug=False)
    for img_c, img_t in data:
        cv2.imshow("rgb",img_c)
        cv2.imshow("thermal",img_t)
        cv2.waitKey(100)


test_dataloader()
