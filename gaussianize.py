
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from nn import Convolution2D


class Gaussianize(nn.Module):
    def __init__(self, c):
        super(Gaussianize, self).__init__()
        self.zeroConv2d = Convolution2D(
            in_dim = c, 
            out_dim= c*2,
            kernel=3, 
            padding=1, 
            last_layer=True, 
            actnorm=False, 
            batchnorm=False
        )

    def forward(self, x_a, x_b):
        log_s_t = self.zeroConv2d.forward(x_a, scale_factor=1.0)
        mean, std = log_s_t[:, 0::2, :, :].contiguous(), log_s_t[:, 1::2, :, :].contiguous()
        x_b = (x_b  - mean) * torch.exp(-std)
        log_det = - std.sum([1,2,3])
        return x_b, log_det #, mean, std

    def inverse(self, z_a, z_b):
        log_s_t = self.zeroConv2d.forward(z_a,scale_factor=1.0)
        mean, std = log_s_t[:, 0::2, :, :].contiguous(), log_s_t[:, 1::2, :, :].contiguous()
        z_b = (z_b + mean) * torch.exp(std)
        log_det = std.sum([1,2,3])
        return z_b, log_det #, mean, std