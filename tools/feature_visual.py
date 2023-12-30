import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import eval
import imageio
import os
from models import TCM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

checkpoint = torch.load("D:\Xml\compression\LIC_TCM0829maskjoint/1021_200epoch/save_path/1checkpoint_latest.pth.tar", map_location='cuda')
net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320, isRGB=True)
net = net.to('cuda')
net.load_state_dict(checkpoint["state_dict"])
net.eval()
print(net)
img_path1 = "D:\Xml\compression\LIC_TCM0829maskjoint\input\MSRS361/vi/00123D.png"
img_path2 = "D:\Xml\compression\LIC_TCM0829maskjoint\input\MSRS361/ir/00123D.png"
img_1 = transforms.ToTensor()(Image.open(img_path1)).to('cuda')
img_2 = transforms.ToTensor()(Image.open(img_path2)).to('cuda')
x1 = img_1.unsqueeze(0)
x2 = img_2.unsqueeze(0)
x_padded_1, padding_1 = eval.pad(x1, 128)
x_padded_2, padding_2 = eval.pad(x2, 128)
# 定义钩子函数，获取指定层名称的特征
activation = {} # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

net.eval()

# 获取layer1里面的bn3层的结果，浅层特征
net.g_a[9].register_forward_hook(get_activation('Conv2d')) # 为layer1中第2个模块的bn3注册钩子
net.h_scale_s[3].register_forward_hook(get_activation('PixelShuffle')) # 为layer1中第2个模块的bn3注册钩子
_ = net(x_padded_1,x_padded_2)

bn3 = activation['Conv2d'] # 结果将保存在activation字典中
bn4 = activation['PixelShuffle'] # 结果将保存在activation字典中
print(bn3.shape)
# 可视化结果，显示前64张
# plt.figure(figsize=(12,12))
for i in range(111,112):
    # plt.subplot(1,1,i+1)
    imageout_path=os.path.join('featuremap_error/', str(i+1)+'.png')
    imageout_pathn=os.path.join('featuremap_norm/', str(i+1)+'.png')
    g_a = bn3[0,i,:,:]
    s = bn4[0,i,:,:]
    m = bn4[0, i+320, :, :]
    error = g_a - m
    norm = error/s
    im = error.cpu().numpy()
    norm1 = norm.cpu().numpy()

    relitu = plt.imshow(s, cmap='Blues', interpolation='nearest',norm=Normalize(vmin=0, vmax=2))
    ax = plt.gca()
    # 添加颜色条
    # 显示热力图
    cbar = plt.colorbar(relitu,ticks=[0, 2])
    cbar.ax.set_position([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.03, ax.get_position().height])
    plt.axis('off')
    plt.show()

    norm_image = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im=norm_image.astype(np.uint8)
    norm_image1 = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normim=norm_image.astype(np.uint8)
    imageio.imwrite(imageout_path, im)
    imageio.imwrite(imageout_pathn, normim)
    # utils.save_images(imageout_path,im)
    # plt.imshow(bn3[0,i,:,:], cmap='gray')
    # plt.axis('off')
# plt.show()




#
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torchvision import transforms
# from torchvision.utils import save_image
# import numpy as np
# from PIL import Image
# from collections import OrderedDict
# import cv2
# import eval
# import imageio
# import os
# from models import TCM
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# checkpoint = torch.load("D:\Xml\compression\LIC_TCM0829maskjoint/1021_200epoch/save_path/1checkpoint_latest.pth.tar", map_location='cuda')
# net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320, isRGB=True)
# net = net.to('cuda')
# net.load_state_dict(checkpoint["state_dict"])
# net.eval()
# print(net)
# img_path1 = "D:\Xml\compression\LIC_TCM0829maskjoint\input\MSRS361/vi/00123D.png"
# img_path2 = "D:\Xml\compression\LIC_TCM0829maskjoint\input\MSRS361/ir/00123D.png"
# img_1 = transforms.ToTensor()(Image.open(img_path1)).to('cuda')
# img_2 = transforms.ToTensor()(Image.open(img_path2)).to('cuda')
# x1 = img_1.unsqueeze(0)
# x2 = img_2.unsqueeze(0)
# x_padded_1, padding_1 = eval.pad(x1, 128)
# x_padded_2, padding_2 = eval.pad(x2, 128)
# # 定义钩子函数，获取指定层名称的特征
# activation = {} # 保存获取的输出
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
#
# net.eval()
#
# # 获取layer1里面的bn3层的结果，浅层特征
# net.g_a[9].register_forward_hook(get_activation('Conv2d')) # 为layer1中第2个模块的bn3注册钩子
# net.h_scale_s[3].register_forward_hook(get_activation('PixelShuffle')) # 为layer1中第2个模块的bn3注册钩子
# _ = net(x_padded_1,x_padded_2)
#
# bn3 = activation['Conv2d'] # 结果将保存在activation字典中
# bn4 = activation['PixelShuffle'] # 结果将保存在activation字典中
# print(bn3.shape)
# # 可视化结果，显示前64张
# # plt.figure(figsize=(12,12))
# for i in range(320):
#     # plt.subplot(1,1,i+1)
#     imageout_path=os.path.join('featuremap_norm/', str(i+1)+'.png')
#     im = bn3[0,i,:,:]
#     im = im.cpu().numpy()
#     norm_image = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     im1=norm_image.astype(np.uint8)
#     imageio.imwrite(imageout_path, im1)
#     # utils.save_images(imageout_path,im)
#     # plt.imshow(bn3[0,i,:,:], cmap='gray')
#     # plt.axis('off')
# # plt.show()
