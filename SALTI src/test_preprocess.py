from preprocess import *
from PIL import Image
def test_resize_and_save_image():
    path=r"D:\asd-pdeng-project-2020-developer\SALTI src\Data\Dataset_V0\images\set00\V000\visible"
    path_resized=r"D:\asd-pdeng-project-2020-developer\SALTI src\Data"
    file="I00000.jpg"
    desired_width=640
    desired_height=640
    resize_and_save_image(path,path_resized,file,desired_width,desired_height)
    resized_image=Image.open(os.path.join(path_resized,file))
    new_width,new_height=resized_image.size
    assert desired_width==new_width
    assert desired_height== new_height