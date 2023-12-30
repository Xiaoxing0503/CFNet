import torch
import torch.nn.functional as F
from torchvision import transforms
from models import CFNet
import warnings
import utils
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())



def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default="save_path/1checkpoint_latest.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--data1", type=str, default="input\MSRS361/vi", help="Path to dataset")
    parser.add_argument("--data2", type=str, default="input\MSRS361\ir", help="Path to dataset")

    parser.add_argument("--output_path", type=str, default="MSRS361", help="Path to output")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=True)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path1 = args.data1
    path2 = args.data2

    img_list1 = []
    for file in os.listdir(path1):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list1.append(file)
    img_list2 = []
    for file in os.listdir(path2):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list2.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = CFNet(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)
    net.eval()
    count = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    net.update()
    for img_name1,img_name2 in zip(img_list1,img_list2):
        img_path1 = os.path.join(path1, img_name1)
        img_path2 = os.path.join(path2, img_name2)
        img_1 = transforms.ToTensor()(Image.open(img_path1)).to(device)
        img_2 = transforms.ToTensor()(Image.open(img_path2)).to(device)
        x1 = img_1.unsqueeze(0)
        x2 = img_2.unsqueeze(0)
        x_padded_1, padding_1 = pad(x1, p)
        x_padded_2, padding_2 = pad(x2, p)

        count += 1
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            s = time.time()
            out_enc = net.compress(x_padded_1,x_padded_2)
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            total_time += (e - s)
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding_1)
            imageout_path = os.path.join(args.output_path, img_name1)

            utils.tensor_save_rgbimage(out_dec["x_hat"], imageout_path)


            num_pixels = x1.size(0) * x1.size(2) * x1.size(3)
            print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
            Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels


    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
