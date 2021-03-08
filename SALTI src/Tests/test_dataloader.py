import unittest
import Dataloader

class MyTestCase(unittest.TestCase):
    def test_something(self):
        path_t = r'D:\KAIST\set00\V000\lwir'
        path_c = r'D:\KAIST\set00\V000\visible'
        data = Dataloader(path_c, path_t, [-1,-1], resize=False)
        for img_c, img_t in data:
            cv2.imshow("demo",img_c)
            cv2.waitKey(100)


#        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
