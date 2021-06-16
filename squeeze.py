import torch.nn as nn

class Squeeze(nn.Module):
    def __init__(self, scale=1):
        super(Squeeze, self).__init__()   

    def forward(self, x):
        b, c, h, w = x.size()
        # Reshape x  
        x = x.view(b, c, h//2, 2, w//2 ,2)
        x = x.permute(0,1,3,5,2,4).contiguous()
        x = x.view(b, c*4, h//2, w//2)
        return x

    def inverse(self, z):
        b, c, h, w = z.size()
        # Reshape z  
        z = z.view(b, c//4, 2, 2, h, w)
        z = z.permute(0,1,4,2,5,3).contiguous()
        z = z.view(b, c//4, h*2, w*2)       
        return z