import torch
import torch.nn as nn 
import torch.nn.functional as F 

class InvConvolution(nn.Module):
    def __init__(self, c):
        super(InvConvolution, self).__init__()
        # Set the device 
        device = 'cpu'

        self.c = c
        w_shape = (c, c)
        # Initialize a random othogonal rotation matrix as a weight matrix 
        self.weights = nn.Parameter(torch.tensor(torch.qr(torch.randn(w_shape))[0], device=device, requires_grad=True))

    def forward(self, x):
        _,_,h,w = x.size()
        log_det =  torch.slogdet(self.weights)[1] * h * w  
        weights = self.weights.view(self.c, self.c, 1, 1)
        x = F.conv2d(x, weights)
        return x, log_det

    def inverse(self, z):
        weights = torch.inverse(self.weights.view(self.c, self.c, 1 ,1))
        z = F.conv2d(z, weights)
        return z

'''
    LU decomposed invertible 1x1 Convultion based on 
    https://github.com/y0ast/Glow-PyTorch/blob/6f71c
    \1def9c8776d2be194ba819177e7db431404/modules.py#L142

'''

class InvConvolutionLU(nn.Module):
    def __init__(self, c):
        super(InvConvolutionLU, self).__init__()
        # Set the device 
        device = 'cpu'

        self.c = c
        w_shape = (c, c)
        weights, _ = torch.qr(torch.randn(w_shape, device=device))
        permute, lower, upper = torch.lu_unpack(*torch.lu(weights))
        diag = torch.diag(upper)
        sign_s = torch.sign(diag)
        log_s = torch.log(torch.abs(diag))
        upper = torch.triu(upper,1)

        self.register_buffer('sign_s', sign_s)
        self.register_buffer('permute', permute)
        self.log_s = nn.Parameter(log_s)
        self.lower = nn.Parameter(lower)
        self.upper = nn.Parameter(upper)
        self.low_mask = torch.tril(torch.ones(w_shape, device=device)-1)
        self.eye = torch.eye(*w_shape, device=device)

    def get_forward_weights(self, x):
        _,_,h,w = x.size() 

        low = self.lower * self.low_mask + self.eye 
        up = self.upper * self.low_mask.transpose(0,1).contiguous()
        up += torch.diag(self.sign_s * torch.exp(self.log_s))
        log_det =  h * w * torch.sum(self.log_s) 

        weight = torch.matmul(self.permute, torch.matmul(low, up))
        return weight.view(self.c ,self.c, 1, 1), log_det 

    def get_inverse_weights(self, z):
        low = self.lower * self.low_mask + self.eye
        up = self.upper * self.low_mask.transpose(0,1).contiguous()
        up += torch.diag(self.sign_s * torch.exp(self.log_s))

        u_inv = torch.inverse(up)
        l_inv = torch.inverse(low)
        p_inv = torch.inverse(self.permute)

        weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
        return weight.view(self.c, self.c, 1, 1)

    def forward(self, x):
        weight, det = self.get_forward_weights(x)
        x = F.conv2d(x, weight)
        # assert not x.mean().isnan(), 'forward permutation is nan'
        # assert not x.mean().isinf(), 'forward permutation is inf'
        return x, det 

    def inverse(self, z):
        weight = self.get_inverse_weights(z)
        z = F.conv2d(z,weight)
        # assert not z.mean().isnan(), 'inverse permutation is nan'
        # assert not z.mean().isinf(), 'inverse permutation is inf'
        return z