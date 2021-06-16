
from gaussianize import Gaussianize
from math import log, pi, exp 
from torch.utils.data.dataloader import DataLoader
from squeeze import Squeeze
import numpy as np
import torchvision
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import matplotlib.pyplot as plt

from split import Split
from normalization import ActNorm
from permutation import InvConvolution, InvConvolutionLU
from couplingLayer import Affine, Additive

class FlowStep(nn.Module):
    def __init__(self, c, affine=True, lu=True, resNetB=False, actnorm=True, batchnorm=False):
        super(FlowStep, self).__init__()
        self.norm = ActNorm(
            # device = device, 
            c = c
        )
        if lu: 
            self.permute = InvConvolutionLU(
                c = c
            )
        else: 
            self.permute = InvConvolution(
                c = c
            )
        if affine: 
            self.coupling = Affine(
                c = c,
                resNetB = resNetB,
                actnorm=actnorm, 
                batchnorm=batchnorm
            )
        else:
            self.coupling = Additive(
                c = c,
                resNetB = resNetB,
                actnorm=actnorm, 
                batchnorm=batchnorm
            )
        
    def forward(self, x):
        x, log_det_norm = self.norm.forward(x)
        x, log_det_permute = self.permute.forward(x)
        x, log_det_couple = self.coupling.forward(x)
        log_det_block = log_det_norm + log_det_permute + log_det_couple
        return x, log_det_block

    def inverse(self, z):
        z = self.coupling.inverse(z)
        z = self.permute.inverse(z)
        z = self.norm.inverse(z)
        return z


# Multi Scale Achitecture 
# Squeeze -> Flow (Activation Normalisation -> 1x1 convolutional permutation -> Affine Coupling) -> Split -> Gaussinization
class Glow(nn.Module):
    def __init__(self, c,  bits=8, nblocks=1, levels=1, affine=True, lu=True, resNetB=False, actnorm=False, batchnorm=False):
        super(Glow, self).__init__()
        self.flows = nn.ModuleList()
        self.bits = bits
        for L in range (levels):
            self.flows.append(Squeeze())
            c = c * 4 
            for k in range(nblocks):
                self.flows.append(FlowStep(c, affine=affine, lu=lu, resNetB=resNetB, actnorm=actnorm, batchnorm=batchnorm))
            if L < levels-1:
                self.flows.append(Split(c))
                c = c // 2
        self.flows.append(Gaussianize(c))
        self.calculated = False

        self.register_buffer('base_mean', torch.zeros(1))
        self.register_buffer('base_std', torch.ones(1))

    def forward(self, x):
        batch, _, _, _ = x.size() 
        Z = [] 
        shapes = []
        log_det = torch.tensor(torch.zeros(batch), device = 'cpu', dtype=torch.float, requires_grad=False) - (log(256) * self.n_pixels)
        for i, module in enumerate(self.flows):
            if module.__class__.__name__ == "Squeeze":
                x = module.forward(x)
            elif module.__class__.__name__ == "FlowStep":
                x, log_det_flowstep = module.forward(x)
                log_det = log_det + log_det_flowstep
            elif module.__class__.__name__ == "Gaussianize":
                x, log_det_gaussian = module.forward(torch.zeros_like(x), x)
                log_det = log_det + log_det_gaussian
            else: 
                x,z, log_det_split = module.split(x, gaus_forward=True)
                log_det = log_det + log_det_split
                shapes.append(z.size())
                Z.append(z)
        shapes.append(x.size())
        Z.append(x)
        return Z, log_det, shapes


    def inverse(self, z):
        x = z[-1]
        k = -2 
        for module in self.flows[::-1]:
            if module.__class__.__name__ == "Gaussianize":
                x, _ = module.inverse(torch.zeros_like(x), x)
            elif module.__class__.__name__ == "Split":
                x = module.concat(x, z[k])
                k = k-1
            else: 
                x = module.inverse(x)
        return x  

    @property
    def priorMultiGaussian(self):
        prior = torch.distributions.Normal(
            loc = self.base_mean, 
            scale= self.base_std
        )
        return prior

    def log_pz(self, z, log_det, bits_per_dim=False):
        log_pz = sum(self.priorMultiGaussian.log_prob(zs).sum([1,2,3]) for zs in z) + log_det
        if bits_per_dim: 
            log_pz = log_pz /(log(2) * self.n_pixels) 
        return log_pz

    def sample(self, shapes, eps=1.0):
            sample = []
            for shape in shapes:
                    z_new = torch.tensor(self.priorMultiGaussian.sample(shape).squeeze(), dtype=torch.float, device='cpu', requires_grad=False) * eps
                    sample.append(z_new)
            return sample

    def reshape_z(self, z, b=8):  
        z_new = [z_new.reshape(b,-1) for z_new in z]
        z_new = torch.cat(z_new, dim=1) 
        return z_new

    def print_model(self):
        for index, module in enumerate(self.flows):
            print("position: ", index, "module: ",module.__class__.__name__)

    def discretize(self, x, n_bins=255):
        # x = (x * n_bins).to(torch.int32)
        # x = x.to(torch.float32) 
        x = x * n_bins 
        if self.bits < 8:
          x = torch.floor(x/2**(8-self.bits))
        x = x / 2**self.bits - 0.5
        return x
    
    def dequantize(self, x, alpha=0.5):
        # x = x.to(torch.float32) 
        x = x + torch.rand_like(x).detach() / self.n_bins 
        x = x 
        return x
        
    def quantize(self, z):
        z = z + 0.5
        return z 
    
    def pixels(self, x):
        _,c,h,w = x.size()
        self.n_pixels = c*h*w
        self.n_bins = 2**self.bits
        self.calculated = True

    def plotImage(image, batch_size=8):
        image = image.cpu().data
        grid = torchvision.utils.make_grid(image, nrow=batch_size)
        grid = grid.detach().numpy()
        plt.figure(figsize=(15,15))
        plt.imshow(np.transpose(grid,(1,2,0)))
        plt.show()
        # image = image.cuda()