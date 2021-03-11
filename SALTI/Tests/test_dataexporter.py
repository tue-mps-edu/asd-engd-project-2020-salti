from DataExporter import *

def test_exporter():
    img = cv2.imread(r'Data\Dataset_V0\images\set00\V000\visible\I00000.jpg')
    classnames = ['car']
    det = Detections([[200, 200, 300, 300]], [0], [0.9])
    out_path = r'D:\Outputs'
    filename = r'I00000'
    E1 = DataExporter('PascalVOC', out_path, classnames)
    E1.export(img.shape, filename, det)
    E2 = DataExporter('YOLO', out_path, classnames)
    E2.export(img.shape, filename, det)

#test_exporter()


# import unittest
# import Dataloader
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         path_t = r'D:\KAIST\set00\V000\lwir'
#         path_c = r'D:\KAIST\set00\V000\visible'
#         data = Dataloader(path_c, path_t, [-1,-1], resize=False)
#         for img_c, img_t in data:
#             cv2.imshow("demo",img_c)
#             cv2.waitKey(100)
#
#
# #        self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()
