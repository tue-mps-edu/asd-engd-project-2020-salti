import os
import cv2
import ntpath

input_formats = ['.png','.jpg','.jpeg']

class DataLoader():
    def __init__(self, path_rgb, path_thermal, debug=False):
        '''
        Class for loading data from a specified folder. An iterable class is created that can be used in a for loop
        Example:
            for file_name, file_ext, img_c, img_t in DataLoader(...):
                foo(img)

        Arguments:
            (str) path to rgb images
            (str) path to thermal images
            (bool) Debug boolean. If true only one out of 60 images is labeled. Additional info is printed.

        Outputs of container iterator
            (str) image filename
            (str) image file extension
            (opencv img) color(rgb) image
            (opencv img) thermal image
        '''

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
        '''
        Create iterator for images

        Example:
            for file_name, file_ext, img_c, img_t in DataLoader(...):
                foo(img)
        '''

        # Loop over list of images stored in class
        for i in range(self.nr_imgs):
            self.count = i

            if self.DEBUG:
                if i%60!=0:
                    continue

            # Get image paths
            path_C = self.imgs_rgb[self.count]              # Color image path
            path_T = self.imgs_thermal[self.count]          # Thermal image path

            # Get file name and extension for use in SALTI
            file = ntpath.basename(path_C)
            file_name, file_ext = os.path.splitext(file)

            # Check if files match
            file_t = ntpath.basename(path_C)
            assert file_t==file, "Files do not match"

            # Read the images
            img_C = cv2.imread(path_C)                      # Read color image
            img_T = cv2.imread(path_T)                      # Read thermal image

            # Assert if images are of same size (thermal twin)
            sz_c = img_C.shape
            sz_t = img_T.shape
            assert sz_c[0] == sz_t[0], "Height of images do not match!"
            assert sz_c[1] == sz_t[1], "Width of images do not match!"

            self.count = self.count + 1  # Update to next image

            # Update progress for progress window
            self.progress = (self.count/self.nr_imgs)

            yield file_name, file_ext, img_C, img_T

