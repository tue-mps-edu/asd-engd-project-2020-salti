import SALTI

def test_validator():
    print("Update the path & labeltype in test_validator before running!")
    results_directory=r'C:\Github\asd-pdeng-project-2020-developer\test_images\outputs\2021.03.26_12h16m40s'
    image_ext='.jpg'
    iou_threshold=0.8
    v = SALTI.Validator(results_directory,image_ext,iou_threshold, 'PaSCalVOC')
    v.complete_Validation()
