import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return torch.abs(sobelx)+torch.abs(sobely)

def L_Grad(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    gradient_A = Sobelxy(image_A_Y)
    gradient_B = Sobelxy(image_B_Y)
    gradient_fused = Sobelxy(image_fused_Y)
    gradient_joint = torch.max(gradient_A, gradient_B)
    Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
    return Loss_gradient

def L_GradRGB(image_A, image_B, image_fused):
    Loss_gradient = F.l1_loss(Sobelxy(image_fused[:, :1, :, :]), torch.max(Sobelxy(image_A[:, :1, :, :]), Sobelxy(image_B[:, :1, :, :])))\
                    + F.l1_loss(Sobelxy(image_fused[:, 1:2, :, :]), torch.max(Sobelxy(image_A[:, 1:2, :, :]), Sobelxy(image_B[:, :1, :, :])))\
                    + F.l1_loss(Sobelxy(image_fused[:, 2:3, :, :]), torch.max(Sobelxy(image_A[:, 2:3, :, :]), Sobelxy(image_B[:, :1, :, :])))
    return Loss_gradient

def L_Int(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    # x_in_max=torch.add(image_A_Y,image_B_Y)/2
    x_in_max = torch.max(image_A_Y, image_B_Y)
    loss_in = F.l1_loss(x_in_max, image_fused_Y)
    return loss_in

def L_IntRGB(image_A, image_B, image_fused):
    # image_fused_Y = image_fused[:, :1, :, :]
    # x_in_max=torch.add(image_A_Y,image_B_Y)/2
    # x_in_max = torch.max(image_A[:, :1, :, :], image_B[:, :1, :, :])
    loss_in = F.l1_loss(torch.max(image_A[:, :1, :, :], image_B[:, :1, :, :]), image_fused[:, :1, :, :]) \
              + F.l1_loss(torch.max(image_A[:, 1:2, :, :], image_B[:, :1, :, :]), image_fused[:, 1:2, :, :]) \
              + F.l1_loss(torch.max(image_A[:, 2:3, :, :], image_B[:, :1, :, :]), image_fused[:, 2:3, :, :])
    return loss_in

def LIntRGBROI(image_A, image_B, image_fused, mask):
    mask = mask[:, :1, :, :]
    loss_in = F.l1_loss(torch.max(image_A[:, :1, :, :]* mask, image_B[:, :1, :, :]* mask) , image_fused[:, :1, :, :] * mask) \
              + F.l1_loss(torch.max(image_A[:, 1:2, :, :]* mask, image_B[:, :1, :, :]* mask) , image_fused[:, 1:2, :, :] * mask) \
              + F.l1_loss(torch.max(image_A[:, 2:3, :, :]* mask, image_B[:, :1, :, :]* mask), image_fused[:, 2:3, :, :] * mask)
    return loss_in

def LIntRGBbg(image_A, image_B, image_fused, mask):
    mask = mask[:, :1, :, :]
    loss_in = F.l1_loss((torch.add(image_A[:, :1, :, :] * mask, image_B[:, :1, :, :] * mask))/2, image_fused[:, :1, :, :] * mask) \
              + F.l1_loss((torch.add(image_A[:, 1:2, :, :] * mask, image_B[:, :1, :, :] * mask))/2, image_fused[:, 1:2, :, :] * mask) \
              + F.l1_loss((torch.add(image_A[:, 2:3, :, :] * mask, image_B[:, :1, :, :] * mask))/2, image_fused[:, 2:3, :, :] * mask)
    return loss_in

def L_GradRGBROI(image_A, image_B, image_fused, mask):
    mask = mask[:, :1, :, :]
    Loss_gradient = F.l1_loss(Sobelxy(image_fused[:, :1, :, :] * mask), torch.max(Sobelxy(image_A[:, :1, :, :] * mask), Sobelxy(image_B[:, :1, :, :] * mask)))\
                    + F.l1_loss(Sobelxy(image_fused[:, 1:2, :, :] * mask), torch.max(Sobelxy(image_A[:, 1:2, :, :] * mask), Sobelxy(image_B[:, :1, :, :] * mask)))\
                    + F.l1_loss(Sobelxy(image_fused[:, 2:3, :, :] * mask), torch.max(Sobelxy(image_A[:, 2:3, :, :] * mask), Sobelxy(image_B[:, :1, :, :] * mask)))
    return Loss_gradient


def L_IntYCbCr(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    # x_in_max=torch.add(image_A_Y,image_B_Y)/2
    x_in_max = torch.max(image_A_Y, image_B_Y)
    loss_in = F.l1_loss(x_in_max, image_fused_Y)
    return loss_in

def L_GradYCbCr(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    gradient_A = Sobelxy(image_A_Y)
    gradient_B = Sobelxy(image_B_Y)
    gradient_fused = Sobelxy(image_fused_Y)
    gradient_joint = torch.max(gradient_A, gradient_B)
    Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
    return Loss_gradient




def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3).cuda()
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window
def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res
# 方差计算
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map_ir=torch.where(map1+mask>0, one, zero)
    map_vi= 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    return res.mean()