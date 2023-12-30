import os
import cv2
import numpy as np
imagePath = 'D:\Xml\Experiment\TNO/U2Fusion'
file = os.listdir(imagePath)
bpps = []
for i in file:
    iPath = os.path.join(imagePath, i)
    image = cv2.imread(iPath)
    [w,h,c] = image.shape
    imagesize = os.path.getsize(iPath)
    bpp = imagesize*8/(w*h)
    bpps.append(bpp)
print(np.mean(bpps))

