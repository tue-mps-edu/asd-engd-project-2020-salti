from Validator import *

def test_validator():
    results_directory=r'C:\Users\20204916\Documents\GitHub\asd-pdeng-project-2020-developer\SALTI\Output\2021.03.14_18h27m52s'
    image_ext='.jpg'
    iou_threshold=0.8
    v = Validator(results_directory,image_ext,iou_threshold)
    v.complete_Validation()
