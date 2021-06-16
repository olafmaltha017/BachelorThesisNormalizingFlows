import torch
import torch.nn as nn 
import torch.nn.functional as F 
from normalization import ActNorm

class Convolution2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1, std=0.05, last_layer=False, actnorm=False, batchnorm=False):
        super(Convolution2D, self).__init__()
        # Set the device 
        device = 'cpu'

        self.conv = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=out_dim, 
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

        self.last_layer=last_layer
        self.actnorm = actnorm, 
        self.batchnorm = batchnorm

        if self.last_layer: 
            self.conv.weight.data.zero_()
            self.conv.bias.data.zero_()
            self.logs = nn.Parameter(torch.zeros(1,out_dim, 1, 1, device=device))
            
        else:  
            self.conv.weight.data.normal_(mean=0.0, std=std)
            if self.actnorm:
                self.actNorm = ActNorm(c=out_dim)
            elif self.batchnorm:
                self.batchnorm = nn.BatchNorm2d(out_dim, affine=True)  
            self.conv.bias.data.zero_()
        self.to(device)
                
    def forward(self, x, scale_factor=3.0):
      if not self.last_layer:
            # x = F.pad(x, [1,1,1,1], value=1)
            x = self.conv(x)
            if self.actnorm:
                x, _ = self.actNorm(x)
            elif self.batchnorm:
                x = self.batchnorm(x)
            # else:
            #     x = F.leaky_relu(x, inplace=False)
            x = F.leaky_relu(x, inplace=False)

      else:
          x = self.conv(x)
          x = x * torch.exp(self.logs * scale_factor)
      return x

class NN(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, actNorm=True, batchNorm=False):
        super(NN, self).__init__()
        # NN shallow ResNet for the affine coupling layer used as the conditioner 
        
        self.conv1 = Convolution2D(
            in_dim=in_dim,
            out_dim=mid_dim,
            std=0.05,
            kernel=3,
            padding=1,
            actnorm=actNorm, 
            batchnorm=batchNorm 
        )
        self.conv2 = Convolution2D(
            in_dim=mid_dim,
            out_dim=mid_dim,
            std=0.05,
            kernel=1, 
            padding=1,
            actnorm=actNorm, 
            batchnorm=batchNorm 
        )
        self.conv3 = Convolution2D(
            in_dim=mid_dim,
            out_dim=out_dim,
            last_layer=True, 
            actnorm=False,
            batchnorm=False,
            kernel=3,
            padding=0
        )

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        return x










# TO DO Alternative ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1, bias=True):
        super(ResNetBlock, self).__init__()
        self.batch_norm_in = nn.BatchNorm2d(in_dim)
        self.w_norm_Conv2D_in = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=in_dim, 
                out_channels=out_dim,
                kernel_size=kernel, 
                stride=stride,
                padding=padding,
                bias=False
            )
        )

        self.batch_norm_out = nn.BatchNorm2d(out_dim)
        self.w_norm_Conv2D_out = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=out_dim, 
                out_channels=out_dim,
                kernel_size=kernel, 
                stride=stride,
                padding=padding, 
                bias=True
            )
        )

    def forward(self, x):
        # with torch.autograd.set_detect_anomaly(True):
        skip_connection = x
        x = self.batch_norm_in(x)
        # x = F.relu(x, inplace=True)
        x = F.relu(x)
        x = self.w_norm_Conv2D_in(x)
        
        x = self.batch_norm_in(x)
        # x = F.relu(x, inplace=True)
        x = F.relu(x)
        x = self.w_norm_Conv2D_out(x)
        x = x + skip_connection 
        return x


class ResNetBatchNorm(nn.Module):
    def __init__(self, in_dim,  mid_dim, out_dim, n_blocks=1):
        super(ResNetBatchNorm, self).__init__()
        self.in_batchnorm = nn.BatchNorm2d(in_dim)
        self.in_conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=mid_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1,
                bias=True
            )
        )
        self.skip_c = nn.utils.weight_norm(
            nn.Conv2d(
                in_channels=mid_dim,
                out_channels=mid_dim, 
                kernel_size=1, 
                stride=1, 
                padding=0,
                bias=True
            )
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(mid_dim, mid_dim) for i in range(n_blocks)]
        )
        # Skip connection Identity matrix 
        self.skipC = nn.ModuleList(
                [nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels=mid_dim,
                        out_channels=mid_dim, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1,
                        bias=True
                    )
            )for i in range(n_blocks)]
        )

        self.out_batchnorm = nn.BatchNorm2d(mid_dim)
        self.out_conv = nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels=mid_dim,
                        out_channels=out_dim, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0,
                        bias=True
                    )
            ) 

    def forward(self, x):
        # with torch.autograd.set_detect_anomaly(True):
        x = self.in_batchnorm(x)
        x *= 2.
        # x = F.relu(x,inplace=False)
        x = F.relu(x)
        x = self.in_conv(x)
        skip_c = self.skip_c(x)

        for resBlock, skipc in zip(self.blocks, self.skipC):
            x = resBlock.forward(x)
            skip_c += skipc(x)

        x = self.out_batchnorm(skip_c)
        # x = F.relu(x, inplace=True)
        x = F.relu(x)
        x = self.out_conv(x)
        return x 