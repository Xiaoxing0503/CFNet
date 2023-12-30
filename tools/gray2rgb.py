'''
单通道->三通道
'''
import os
import cv2
import numpy as np
import PIL.Image as Image
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
img_path='D:\Xml\Experiment\TNO/111_vis_original/'
save_img_path='D:\Xml\Experiment\TNO/111_vis_original_3/'
for img_name in os.listdir(img_path):
   image=Image.open(img_path+img_name)
   if len(image.split())==1: #查看通道数
       print(len(image.split()))
       print(img_path+img_name)
       img = cv2.imread(img_path+img_name)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       img2 = np.zeros_like(img)
       img2[:,:,0] = gray
       img2[:,:,1] = gray
       img2[:,:,2] = gray
       cv2.imwrite(save_img_path+img_name, img2)
       image=Image.open(save_img_path+img_name)
       print(len(image.split()))
   else:
       image.save(save_img_path+img_name)

'''
单通道->三通道
'''
#img_src = np.expand_dims(img_src, axis=2)
#img_src = np.concatenate((img_src, img_src, img_src), axis=-1)
