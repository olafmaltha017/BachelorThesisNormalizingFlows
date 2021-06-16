import torch
import torch.nn as nn
from gaussianize import Gaussianize 

class Split(nn.Module):
    def __init__(self, c):
        super(Split, self).__init__()
        self.c = c
        self.gausianize = Gaussianize(c//2)

    def split(self, x, gaus_forward=False):
        x_a = x[:, :self.c//2, :, :]
        x_b = x[:, self.c//2:, :, :]
        if gaus_forward:
            x_b, log_det = self.gausianize.forward(x_a, x_b)
            return x_a, x_b, log_det 
        return x_a, x_b

    def concat(self, x_a, x_b, gaus_inv=False):
        if gaus_inv:
            x_b, _ = self.gausianize.inverse(x_a, x_b)
            x = torch.cat([x_a, x_b], dim=1)
            return x
        else:
            x = torch.cat([x_a, x_b], dim=1)
        return x 