
import os
import sys
#singan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
#print(singan_dir)
#sys.path.append(singan_dir)


import SinGAN

import torch

def load_from_dir(dir):
    
    
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please check your checkpoint directory')
    return Gs,Zs,reals,NoiseAmp