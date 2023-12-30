import argparse
import math
import time
import random
import sys
import loss
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim
import RGB_YCrCb

from models import CFNet
# from torch.utils.tensorboard import SummaryWriter
import os
# from data.dataloder import Dataset as D

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type



    def forward(self, output, target1, target2, mask,device):
        N, _, H, W = target1.size()
        out = {}
        num_pixels = N * H * W
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask > 0, one, zero)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["pixel_loss_ROI"] = loss.LIntRGBROI(target1, target2, output["x_hat"], mask)
        out["grad_loss_ROI"] = loss.L_GradRGBROI(target1, target2, output["x_hat"], mask)
        out["pixel_loss_bg"] = loss.LIntRGBROI(target1, target2, output["x_hat"], 1 - mask)
        out["grad_loss_bg"] = loss.L_GradRGBROI(target1, target2, output["x_hat"], 1 - mask)
        W_roi = torch.mean(mask)
        W_bg = torch.mean(1 - mask)
        out["ROI"] = (out["pixel_loss_ROI"] + out["grad_loss_ROI"]) * W_roi / (W_roi + W_bg)
        out["bg"] = (out["pixel_loss_bg"] + out["grad_loss_bg"]) * W_bg / (W_roi + W_bg)
        out["content_loss"] = (out["ROI"] * 2)*500

        out["loss"] = self.lmbda * out["content_loss"] + out["bpp_loss"]
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
         lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_dataloader1, train_dataloader2, optimizer, aux_optimizer, epoch, clip_max_norm, mask, type='mse'):
    model.train()
    device = next(model.parameters()).device

    for i,d1,d2,mask in zip(range(len(train_dataloader1)),train_dataloader1,train_dataloader2,mask):

        d1 = d1.to(device)
        d2 = d2.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d1, d2)

        out_criterion = criterion(out_net, d1, d2, mask,device)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1 == 0:
            print(
                f"{time.ctime()}"
                f"\tTrain epoch {epoch}: ["
                f"{i*len(d1)}/{len(train_dataloader1.dataset)}"
                f" ({100. * i / len(train_dataloader1):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tROI_Loss: {out_criterion["ROI"].item():.3f} |'
                f'\tpixel_loss_ROI: {out_criterion["pixel_loss_ROI"].item():.3f} |'
                f'\tgrad_loss_ROI: {out_criterion["grad_loss_ROI"].item():.3f} |'
                f'\tpixel_loss_bg: {out_criterion["pixel_loss_bg"].item():.3f} |'
                f'\tgrad_loss_bg: {out_criterion["grad_loss_bg"].item():.3f} |'
                f'\tbg loss: {out_criterion["bg"].item():.3f} |'
                # f'\tssim_loss: {out_criterion["ssim_loss"].item():.3f} |'
                f'\tContent loss: {out_criterion["content_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def save_checkpoint(state, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-d", "--dataset", type=str, default="D:/Xml/datasets/MSRS-main/train", help="Training dataset"
    )
    parser.add_argument('--A_dir', type=str, default='vi',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='ir',
                        help='input test image name')
    parser.add_argument('--mask', type=str, default='bi', help='input test image name')
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    # parser.add_argument("--checkpoint", type=str, default="D:\Xml\compression\LIC_TCM0829maskjoint\save_path/1checkpoint_latest.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, default='./save_path', help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    A_dir = os.path.join(args.dataset, args.A_dir)
    B_dir = os.path.join(args.dataset, args.B_dir)
    mask_dir = os.path.join(args.dataset, args.mask)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms1 = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()])

    train_transforms2 = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor(), transforms.Grayscale()]
    )



    A_dataset = ImageFolder(A_dir, split="train", transform=train_transforms1)
    B_dataset = ImageFolder(B_dir, split="train", transform=train_transforms2)
    mask_daset = ImageFolder(mask_dir, split="train", transform=train_transforms2)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader1 = DataLoader(
        A_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    train_dataloader2 = DataLoader(
        B_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    train_mask = DataLoader(
        mask_daset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    net = CFNet(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320, isRGB=True)
    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net) #数据并行

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader1,
            train_dataloader2,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            train_mask,
            type
        )

        lr_scheduler.step()


        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                # True,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])