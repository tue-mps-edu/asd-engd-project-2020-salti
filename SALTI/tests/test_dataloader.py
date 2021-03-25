from DataLoader import *

def test_dataloader():
    path_t = r'D:\KAIST\set00\V000\lwir'
    path_c = r'D:\KAIST\set00\V000\visible'
    data = DataLoader(path_c, path_t, output_size=[320,320], debug=True)
    for img_c, img_t in data:
        cv2.imshow("rgb",img_c)
        cv2.imshow("thermal",img_t)
        cv2.waitKey(100)
