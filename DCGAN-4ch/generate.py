from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', help='path to check point', default="/work/vajira/DL/DCGAN/out/netG_epoch_500.pth")
parser.add_argument('--dest', help='path to save data', default="/work/vajira/DATA/DCGAN_polyps")
parser.add_argument("--nc", type=int, default=4, help="number of channels in input images")
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument("--nsamples", type=int, default=10, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--save_option", default="image_and_mask", help="Options to svae output, image_only, mask_only, image_and_mask", choices=["image_only","mask_only", "image_and_mask"])
    
#parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc=opt.nc

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(opt.ngpu).to(device)
#netG.apply(weights_init)
#if opt.netG != '':
netG.load_state_dict(torch.load(opt.ckpt))
netG.eval()
print("Checkpint is loaded succesfully..!")

dest = opt.dest
os.makedirs(dest, exist_ok=True)


#print(g_img.shape)


for i in tqdm(range(opt.nsamples)):
    noise = torch.randn(1, nz, 1, 1, device=device)
    g_img = netG(noise)[0]
    g_mask = g_img.add(1).mul(0.5)[-1, :, :].expand(3, -1, -1)
    g_img = g_img.add(1).mul(0.5)[0:3, :, :]

    if opt.save_option == "image_and_mask":
        vutils.save_image(g_img, 
            os.path.join(dest, '%d_img.png'%(i)))#, normalize=True, range=(-1,1))
        vutils.save_image(g_mask, 
            os.path.join(dest, '%d_mask.png'%(i)))#, normalize=True, range=(-1,1))

    elif opt.save_option == "image_only":
        vutils.save_image(g_img, 
            os.path.join(dest, '%d_img.png'%(i)))#, normalize=True, range=(-1,1))
                    
    elif opt.save_option == "mask_only":
        vutils.save_image(g_mask, 
            os.path.join(dest, '%d_mask.png'%(i)))#, normalize=True, range=(-1,1))
    else:
        print("wrong choise to save option.") 

