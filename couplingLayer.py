import torch
import torch.nn as nn 
import torch.nn.functional as F 

from nn import NN, ResNetBatchNorm
from split import Split

class Additive(nn.Module):
    def __init__(self, c, eps=1e-6, resNetB=False, actnorm=False, batchnorm=False):
        super().__init__()
        self.c = c
        if resNetB:
            self.nn = ResNetBatchNorm(
                in_dim = self.c//2,
                mid_dim = self.c, 
                out_dim = self.c//2,
                n_blocks=3
            )
        else:
            self.nn = NN(
                in_dim = self.c//2,
                mid_dim = self.c, 
                out_dim = self.c//2,
                actNorm = actnorm,
                batchNorm = batchnorm,
            )
        self.split = Split(
            self.c
        )

    def forward(self, x):
        b,_,_,_ = x.size()
        # Partitioning the data into two halfs
        # x_b is the conditioner 
        x_a, x_b = self.split.split(x)
        transform = self.nn.forward(x_b)
        x_a = (x_a + transform)
        x = self.split.concat(x_a, x_b)
        log_det = torch.tensor(0, dtype=torch.float, requires_grad=False)
        return x, log_det

    def inverse(self, z):
        z_a, z_b = self.split.split(z)
        transform = self.nn.forward(z_b)
        z_a = (z_a - transform)
        z = self.split.concat(z_a, z_b)
        return z 

class Affine(nn.Module):
    def __init__(self, c, eps=1e-6, resNetB=False, actnorm=False, batchnorm=False):
        super().__init__()
        self.c = c
        self.eps = eps
        if resNetB:
            self.nn = ResNetBatchNorm(
                in_dim = self.c//2,
                mid_dim = self.c, 
                out_dim = self.c,
                n_blocks=10
            )
        else: 
            self.nn = NN(
                in_dim = self.c//2,
                mid_dim = self.c, 
                out_dim = self.c,
                actNorm = actnorm,
                batchNorm = batchnorm,
            )
        self.split = Split(
            self.c
        )

    def res_sigmoid(self, scale, top=1, bottom=0.5):
        return bottom + (top - bottom)/(1 + torch.exp(scale))
        
    def forward(self, x, sigm=False):
        b,_,_,_ = x.size()
        # Partitioning the data into two halfs
        # x_b is the conditioner 
        x_a, x_b = self.split.split(x)
        # x_a, x_b = x.chunk(2,1)
        # An Affine transformation using a shallow neural network ResNet
        log_s_t = self.nn(x_b)
        scale, transform =  log_s_t[:, 0::2, :, :].contiguous(), log_s_t[:, 1::2, :, :].contiguous()
        if sigm:
            scale = torch.sigmoid(scale + 2.0)
        else:
            scale = self.res_sigmoid(scale)
        x_a = x_a + transform
        x_a = scale * x_a
        x = self.split.concat(x_a, x_b)
        log_det = torch.log(scale).view(b,-1).sum(-1)
        return x, log_det

    def inverse(self, z, sigm=False):
        z_a, z_b = self.split.split(z)
        log_s_t = self.nn.forward(z_b)
        scale, transform = log_s_t[:, 0::2, :, :].contiguous(), log_s_t[:, 1::2, :, :].contiguous()
        if sigm:
            scale = torch.sigmoid(scale + 2.0)
        else:
            scale = self.res_sigmoid(scale)
        z_a = z_a / scale
        z_a = z_a - transform
        z = self.split.concat(z_a, z_b)
        return z