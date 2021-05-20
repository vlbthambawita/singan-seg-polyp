import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from singan_polyp_aug.config import get_arguments
from singan_polyp_aug.SinGAN.manipulate import *
from singan_polyp_aug.SinGAN.training import *
from singan_polyp_aug.SinGAN.imresize import imresize
import singan_polyp_aug.SinGAN.functions as functions

from singan_polyp_aug.prepare_requirements import prepare_checkpoints

def generate_from_single_checkpoint(out_dir:str, 
                                    checkpoint_path:str, 
                                    num_samples:int=1, 
                                    gen_start_scale:int=5,
                                    mask_post_processing:bool=True) -> None:
    ''' A function to generate synthetic polyp and correspondign mask from a given checkpoint path.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    checkpoint_path: str
        A path to a downloaded checkpoint. To get paths, you have to run prepare_requirements.prepare_checkpoints() function.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''
    
    checkpoint_id = str(checkpoint_path).split("/")[-1] # get the id of the checkpoint 
    checkpoint_dir = os.path.join("/", *str(checkpoint_path).split("/")[:-2])
    #print("chk_dir:", checkpoint_dir)
    image_id = str(checkpoint_id)
    #print("img_di:", image_id)
    #gen_start_scale = 5
    
    parser = get_arguments()
    #parser.add_argument('--input_dir', help='input image dir', default='/work/vajira/DATA/hyper_kvasir/data_new/segmented_train_val/data/img_and_mask_together')
    parser.add_argument('--input_name', help='input image name', default="")
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='random_samples')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    opt = parser.parse_args("")
    opt = functions.post_config(opt)
    
    
    
    opt.input_name = str(checkpoint_id)
    opt.gen_start_scale = gen_start_scale
    opt.nc_z = 4
    opt.nc_im = 4
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.input_name = opt.input_name + ".jpg" # to match with original shirnking function
    opt.out = out_dir
    opt.device
    
    # set output information
    opt.dir2save = out_dir
    opt.checkpoint_id = checkpoint_id
    opt.mask_post_processing = mask_post_processing
    
    # checkpoint paths
    Gs_path = f"{checkpoint_path}/Gs.pth"
    noise_path = f"{checkpoint_path}/NoiseAmp.pth"
    reals_path = f"{checkpoint_path}/reals.pth"
    z_path = f"{checkpoint_path}/Zs.pth"
    
    # Loading checkpoints
    Gs = torch.load(Gs_path, map_location=opt.device)
    reals = torch.load(reals_path, map_location=opt.device)
    Zs = torch.load(z_path, map_location=opt.device)
    NoiseAmp = torch.load(noise_path, map_location=opt.device)

    real = functions.read_image_from_path(os.path.join(checkpoint_dir, "real_images", image_id + ".jpg"), opt)
    #print(opt)
    #print("====opt-after======")
    functions.adjust_scales2image(real, opt)
    #print(real.shape)
    #print(opt)
    
    #in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
    # Generate and save
    out = SinGAN_generate_clean(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale, num_samples=num_samples)
    
    #print(opt.mode)
    
    return None



def generate_from_multiple_checkpoints(out_dir:str, checkpoint_paths:list, *args, **kwargs)-> None:
    ''' A function to generate synthetic polyp and correspondign mask from a given list of checkpoint paths.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    checkpoint_paths: list
        A path list to downloaded checkpoints. To get paths, you have to run prepare_requirements.prepare_checkpoints() function.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''
    
    print(len((kwargs)))
    
    for chk_path in tqdm(checkpoint_paths):
        generate_from_single_checkpoint(out_dir, chk_path, *args, **kwargs)
        
    return None


def generate_simple(out_dir:str, chk_dir:str, *args, **kwargs)-> None:
    ''' A function to generate synthetic polyp and correspondign mask from all downloaded checkpoint paths.

    Parameters
    ----------
    out_dir: str
        A path to save output data.
    chk_dir: str
        The path to checkpoint directory. If the directory does not have downloaded checkpoints, this function will download them.
    num_samples: int
        Number of random samples to generate from the given checkpoint. Default=1.
    gen_start_scale: int
        Predefined scales used in SinGAN training. You can use values between 0-9. Value 0 generates more random samples and value 9 generate sampels which are 
        very close to the training image.
    mask_post_processing: bool
        Whether the generated mask should be post processed or not. If True, generates mask is cleaned to have only 0 and 255. 
    
    Returns
    ------
    None
        This function does not have a return. 

    '''

    
    paths, _ = prepare_checkpoints(chk_dir)
    
    time.sleep(2) # to get clear output for tqdm
    
    generate_from_multiple_checkpoints(out_dir, paths, *args, **kwargs)
    
    return None
