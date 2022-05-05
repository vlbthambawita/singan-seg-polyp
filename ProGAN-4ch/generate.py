import torch
from progan_modules_4ch import Generator
import argparse
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def imsave(inp, path, cmap=None):
    """Imshow for Tensor."""
    inp = denorm(inp)
    
    if cmap==None:
        inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
        plt.imsave(path, inp[:,:, :], vmin=0, vmax=1, cmap=cmap)
    else:
        inp = inp.detach().cpu().numpy()
        plt.imsave(path, inp[:, :], vmin=0, vmax=1, cmap=cmap)


def imshow(inp):
    """Imshow for Tensor."""
    inp = denorm(inp)
    print(inp.max())
    print(inp.min())
    inp = inp.detach().cpu().numpy().transpose((1, 2, 0))
 
    
    plt.imshow(inp[:,:, :], vmin=0, vmax=1)
    #if title is not None:
    #    plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

def load_and_generate(args):
    generator = Generator(in_channel=args.channel, input_code_dim=args.z_dim, pixel_norm=args.pixel_norm, tanh=args.tanh, img_channels= args.img_channels).to(args.device)
    generator.load_state_dict(torch.load(args.path_G))
    model =generator.eval()
    
    # prepare output dir
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(args.num_fakes)):
        gen_z = torch.randn(1, args.z_dim).to(args.device)
        fake = generator(gen_z, step=args.step, alpha=args.alpha)[0]
        
        
        if args.save_option == "image_and_mask":
            imsave(fake[0:3, :, :], f"{output_dir}/{i}_img.png", cmap=None)
            imsave(fake[3, :, :], f"{output_dir}/{i}_mask.png", cmap="gray")
            
        elif args.save_option == "image_only":
            imsave(fake[0:3, :, :], f"{output_dir}/{i}_img.png", cmap=None)
            
        elif args.save_option == "mask_only":
            imsave(fake[3, :, :], f"{output_dir}/{i}_mask.png", cmap="gray")
            #imshow(fake[3:, :, :])
            #imshow(fake[0:3, :, :])
            
        else:
            print("wrong choise to save option.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Progressive GAN")
    parser.add_argument('--path_G', default="/work/vajira/DL/Progressive-GAN-pytorch-4ch/trial_test1_2022-02-23_8_26/checkpoint/300000_g.model")
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--img_channels', default=4, help="Number of channels in input data., for rgb images=3, gray=1 etc.")
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    parser.add_argument("--step", default=6, help="step to generate fake data. # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 256")
    parser.add_argument("--alpha", default=1, help="Progressive gan parameter to set, 0 or 1")

    #Saving location
    parser.add_argument("--output_dir", default="/work/vajira/DATA/progressive_gan_polyps/generated_1k_set_4", help="locations to save generated images")
    parser.add_argument("--save_option", default="image_only", help="Options to svae output, image_only, mask_only, image_and_mask", choices=["image_only","mask_only", "image_and_mask"])
    parser.add_argument("--num_fakes", default=1000, help="Number of fakes to generate.", type=int)

    args = parser.parse_args()

    args.device = torch.device("cuda:%d"%(args.gpu_id)) 

    load_and_generate(args) 

