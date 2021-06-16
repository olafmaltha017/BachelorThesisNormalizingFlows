# Standaard imports
import os
import math
import time
import numpy as np
import zipfile

# Pytorch
import torch
from torch._C import device
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.utils as utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
## Progress bar
from tqdm.notebook import tqdm

# Google colab
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)
# PyTorch Lightning
# try

# Plotting imports 
import matplotlib.pyplot as plt

class DefaultLoader(nn.Module):
    def __init__(self, root):
        super(DefaultLoader, self).__init__()
        self.root_path = root
        if not os.path.exists(root):
            os.mkdir(root)

    def dataloader(self, batch_size=1, split=False, dataset='MNIST'):
        if dataset == 'MNIST':
            train_set=torchvision.datasets.MNIST(
                root= self.root_path +'_MNIST'
                ,train=True
                ,download=True
                ,transform=transforms.Compose([
                    # transforms.Pad(2),                                        
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: x + torch.rand_like(x)/2**8),  
                    # transforms.Lambda(lambda x: x.expand(3,-1,-1))
                ]))
        elif dataset == 'FashionMNIST':
             train_set = torchvision.datasets.FashionMNIST(
                root=self.root_path+'_fashionMNIST'
                ,train=True
                ,download=True
                ,transform=transforms.Compose([
                    # transforms.Pad(2),                                        
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    # transforms.Lambda(lambda x: x + torch.rand_like(x)/2**8),  
                    # transforms.Lambda(lambda x: x.expand(3,-1,-1))
                ])
            )
        else:
             train_set = torchvision.datasets.CIFAR10(
                root=self.root_path+'_CIFAR10'
                ,train=True
                ,download=True
                ,transform=transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            )

        # Train data
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False
        )
        if split:
            train_set, val_set = torch.utils.data.random_split(
                train_set, 
                [5000,1000]
                )
            # # Validation data
            val_loader = DataLoader(
                val_set, 
                batch_size=batch_size, 
                shuffle=False, 
                drop_last=False
            )
            return train_loader, val_loader
        else:
            return train_loader 