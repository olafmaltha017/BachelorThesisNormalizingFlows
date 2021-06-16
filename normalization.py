import torch
import torch.nn as nn 
import torch.nn.functional as F 

class ActNorm(nn.Module):
    def __init__(self, c, scale_factor=1.0):
        super(ActNorm, self).__init__()
        # Set the device 
        device = 'cpu'
        # Activation normalization z = x * scale + bias 
        size = (1, c, 1, 1)
        self.scale_factor=scale_factor
        self.bias = nn.Parameter(torch.zeros(size, dtype=torch.float, device=device, requires_grad=True))
        self.scale = nn.Parameter(torch.ones(size, dtype=torch.float, device=device, requires_grad=True))
        self.initalized = False


    def initialize(self, x):
        # Data dependent initialization
        with torch.no_grad():
            bias = torch.mean(x.clone(), dim=[0,2,3], keepdim=True)
            var = torch.std((x.clone() - bias)**2, dim=[0,2,3], keepdim=True)
            scale = (self.scale_factor / (torch.sqrt(var) + 1e-6)).log()
            self.scale.data.copy_(scale.data)
            self.bias.data.copy_(-bias.data)
            self.initalized = True


    def forward(self, x):
        _,_,h,w = x.size()
        if self.initalized == False:
            self.initialize(x)
        x = x + self.bias
        x = x * torch.exp(self.scale)
        log_det = h * w * self.scale.view(-1).sum()
        return x, log_det
      

    def inverse(self, z):
        z = z * torch.exp(-self.scale)
        z = z - self.bias
        return z 