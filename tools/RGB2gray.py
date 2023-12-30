'''
单通道->三通道
'''
import os
import cv2
import numpy as np
import PIL.Image as Image
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
img_path='D:\Xml\compression\LIC_TCM0829maskjoint\outputTNO40/'
for img_name in os.listdir(img_path):
   image=Image.open(img_path+img_name)
   print(img_path+img_name)
   img = cv2.imread(img_path+img_name)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   cv2.imwrite(img_path+img_name, gray)


