'''
This file contains all the functions that are used for preprocessing the images
'''

#function to take an input image, resize it according to the desired height and width and save it to a specific directory
def resize_and_save_image(path,path_resized,file,desired_width,desired_height):
    original_image = Image.open(os.path.join(path,file))
    resized_image = original_image.resize((desired_width,desired_height), Image.ANTIALIAS)
    resized_image.save(os.path.join(path_resized,file))

def preprocess(image):
    '''
    Insert the preprocessing code
    :param image:
    :return:
    '''
    return image
