import os
import cv2
import ntpath

input_formats = ['.png','.jpg','.jpeg']

class DataLoader():
    def __init__(self, path_rgb, path_thermal, output_size=[-1,-1], debug=False):
        self.DEBUG = debug                      # Debug flag for skipping images
        self.count = 0                          # Logging the current image
        self.progress = 0                       # for progress bar
        self.path_rgb = path_rgb                # path to rgb directory
        self.path_thermal = path_thermal        # path to thermal directory

        # Only directories are allowed as input
        assert(os.path.isdir(path_rgb) and os.path.isdir(path_thermal))

        # Get the file list
        files_rgb = [os.path.abspath(os.path.join(path_rgb, p)) for p in os.listdir(path_rgb)]
        files_thermal = [os.path.abspath(os.path.join(path_thermal, p)) for p in os.listdir(path_thermal)]

        # Get a list of the images
        self.imgs_thermal = [x for x in files_thermal if os.path.splitext(x)[-1].lower() in input_formats]
        self.imgs_rgb = [x for x in files_rgb if os.path.splitext(x)[-1].lower() in input_formats]

        img_temp = cv2.imread(files_rgb[0])
        self.img_size = img_temp.shape

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

            file = ntpath.basename(path_C)
            file_name, file_ext = os.path.splitext(file)

            #file_name, file_ext = os.path.splitext(path_C)

            #assert(len(file_name)<10)

            if self.DEBUG:
                print('color image: \t'+path_C)
                print('thermal image: \t'+path_T)

            img_C = cv2.imread(path_C)                      # Read color image
            img_T = cv2.imread(path_T)                      # Read thermal image

            self.count = self.count + 1                     # Update to next image
            self.progress = (self.count/self.nr_imgs)   # Update progress


            yield file_name, file_ext, img_C, img_T

