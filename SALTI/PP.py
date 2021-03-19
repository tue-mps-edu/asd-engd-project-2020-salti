import cv2
import numpy as np
# import os
# import sys
# def find_energy(img_gray):
#     # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_gray = img_gray.astype('float64')
#     size = np.shape(img_gray)
#     energy = np.sqrt(np.mean(img_gray**2))
#     return energy
# def adaptive_histogram_enhance(img_gray_received):
#     energy = find_energy(img_gray_received)
#     add_factor = int(120 - energy)
#     img_corrected = img_gray_received.copy() * 1.0
#     img_corrected = img_corrected + add_factor
#     img_corrected[img_corrected >= 255] = 255
#     img_corrected[img_corrected <= 0] = 0
#     img_corrected = img_corrected.astype(np.uint8)
#     return img_corrected
# def adaptive_sharpen(img_gray_received):
#     img_sharpened = img_gray_received.copy() * 1.0
#     blurred = cv2.GaussianBlur(img_sharpened, (5, 5), 2) * 1.0
#     img_sharpened = 2.0 * img_sharpened - 1.6 * blurred
#     sharp_energy = find_energy(img_sharpened)
#     sharp_factor = 100/sharp_energy
#     img_sharpened = img_sharpened * sharp_factor
#     img_sharpened[img_sharpened <= 0] = 0
#     img_sharpened[img_sharpened >= 255] = 255
#     img_sharpened = img_sharpened.astype(np.uint8)
#     return img_sharpened
# #img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
# #path = r'C:\Users\20204922\PycharmProjects\Yolov3_thermal\Object-Detection-on-Thermal-Images\data\samples'
# #images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
# ROOT_DIR = r'D:\PDEng2020-2022\Block 2\In-House project\ASD SCRUM\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_NIGHT'
# print(ROOT_DIR)
# sys.path.append(ROOT_DIR)
# IMAGE_DIR = os.path.join(ROOT_DIR, 'To_filter')
# print(IMAGE_DIR)
# file_names = next(os.walk(IMAGE_DIR))[2]
# for file_name in file_names:
#     img_rgb = cv2.imread(os.path.join(IMAGE_DIR, file_name))
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#     img_gray = img_gray.astype(np.uint8)
#     #img_corrected = adaptive_histogram_enhance(img_gray)
#     #cv2.imshow("original", img_gray)
#     #cv2.imshow("corrected", img_corrected)
#     sharpened = adaptive_sharpen(img_gray)
#     cv2.imshow("sharpened image", sharpened)
#     Pathtosaveimage = os.path.join(r'D:\PDEng2020-2022\Block 2\In-House project\ASD SCRUM\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_NIGHT\test124', file_name)
#     cv2.imwrite(Pathtosaveimage, sharpened)
#
#

#compare pp's

a = cv2.imread(r'D:\PDEng2020-2022\Block 2\In-House project\ASD SCRUM\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_NIGHT\saved_pp\saved.jpg')
# a= np.reshape(a,(1,))
# print(a)
b = cv2.imread(r'D:\PDEng2020-2022\Block 2\In-House project\ASD SCRUM\asd-pdeng-project-2020-developer\SALTI\Data\KAIST_NIGHT\saved_pp\I00000.jpg')
c = a-b
print(c[c>0])

print(np.array_equal(a,b))