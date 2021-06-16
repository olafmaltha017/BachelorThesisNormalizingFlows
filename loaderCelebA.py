# Standaard imports
import os
import zipfile

# Pytorch
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torch.utils as utils
from torchvision import datasets, transforms, utils


class CelebALoader(nn.Module):
    def __init__(self,  folder_path, display_features=True):
        super(CelebALoader, self).__init__()
        # Create the directory 
        self.folder_path = folder_path
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            os.makedirs(folder_path)

        with zipfile.ZipFile("celeba.zip","r") as zip_ref:
            zip_ref.extractall(self.folder_path)

    def sample_data(self,batch_size=1, image_size=64, split=False):
        image_names = os.listdir(self.folder_path)
        print(len(image_names))
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
            ]
        )
        dataset = datasets.ImageFolder(self.folder_path, transform=transform)
        if split:
            dataset = torch.utils.data.random_split(dataset, [90,len(dataset)-90])[0]
        train_loader = utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return train_loader
    