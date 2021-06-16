# Standaard imports
import os
import argparse
import json 
from time import sleep
from math import log, pi, exp 
import random 


import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils as utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from loaderCelebA import CelebALoader
from loader import DataLoader

from torch import optim
from tqdm import tqdm
from ipywidgets import IntProgress

from torch.utils.tensorboard import SummaryWriter

# Plotting imports 
import matplotlib.pyplot as plt

from flows import Glow
from loader import DefaultLoader

def get_current_lr(optimizer, group_idx, parameter_idx):
    group = optimizer.param_groups[group_idx]
    p = group['params'][parameter_idx]

    beta1, _ = group['betas']
    state = optimizer.state[p]

    bias_correction1 = 1 - beta1 ** state['step']
    current_lr = group['lr'] / bias_correction1
    return current_lr


def train(device, parser, writer):
  args = parser.parse_args()
  root = f'./Data'
  save_path = f'./checkpoints/checkpoint.pt'
  args_path = f'./checkpoints/commandline_args.txt'

  with open(args_path, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

  with open(args_path, 'r') as f:
    args.__dict__ == json.load(f) 

  if not os.path.exists(root):
      os.mkdir(root)

  if args.print_dict:
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print('Using device: ', device, '\n')

  data_loader = DefaultLoader(root)
  if args.split:
    train_loader, val_loader = data_loader.dataloader(args.batch_size, args.split, args.dataset)
  else:
    train_loader = data_loader.dataloader(args.batch_size, args.split, args.dataset)
  glow = Glow(args.c, args.n_bits, args.n_blocks, args.levels, args.affine, args.lu, args.resNet, args.actnorm, args.batchnorm)
  
  optimizer = optim.Adam(
      params=glow.parameters(), 
      lr=args.learning_rate
  )

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',
      patience = 500,   
      verbose=True,
      min_lr=1e-8
  )
  
  if os.path.exists(save_path):
    print('loading model...')
    checkpoint = torch.load(save_path)
    glow.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['schedular_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
  
  global_steps = 0
  running_loss = 0.0
  # Training loop 
  for epoch in range(1,args.epoch):
    epoch_loss = []
  
    index = 0
    # with tqdm(train_loader, unit="batch") as tepoch:
    for i, batch in enumerate (train_loader):
      x = batch[0]
      x = x.to(device)
      x = x.requires_grad_(False)
      if not glow.calculated:
        glow.pixels(x)
      x = glow.discretize(x)
      x = glow.dequantize(x)
      x = x - 0.5
      x, log_det, shapes = glow.forward(x)
      bpd = glow.log_pz(x, log_det, True)
      
      optimizer.zero_grad()
      
      loss = -bpd.mean(0)
      epoch_loss.append(loss)
      scheduler.step(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(glow.parameters(),2)
      torch.nn.utils.clip_grad_norm_(glow.parameters(), 50)
      
      optimizer.step()
      # tepoch.set_description(f" Epoch {epoch}")
      z_std = [0., 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
      if index % 250 == 0:
        with torch.no_grad():
          rand_idx = random.randint(0, len(z_std)-1)
          eps = z_std[rand_idx]
          sample = glow.sample(shapes=shapes, eps=eps)
          sample = glow.inverse(sample)
          sample = glow.quantize(sample)
          sample_grid = utils.make_grid(sample[:10], nrow=15)
          sample_grid = sample_grid.detach().numpy()
          # writer.add_image('sample'+str(global_steps), sample_grid)
          utils.save_image(
            sample,
            f'./samples/sample{str(epoch+index+1).zfill(6)}.png',
            normalize=True, 
            nrow=10,
            range=(-0.5,0.5),
          )
          if index % 500 == 0:
            x = glow.inverse(x)
            x = glow.quantize(x)
            check_grid = utils.make_grid(x[:10], nrow=15)
            check_grid = check_grid.detach().numpy()
            # writer.add_image('check'+str(global_steps), check_grid)
            utils.save_image(
              x,
              f'./check/checked{str(epoch+index+1).zfill(6)}.png',
              normalize=True, 
              nrow=10,
              range=(-0.5,0.5),
            )

        writer.add_scalar('training loss', running_loss/100,  epoch * global_steps + index)
      group_idx, param_idx = 0, 0
      current_lr = get_current_lr(optimizer, group_idx, param_idx)
      writer.add_scalar('learning rate ', current_lr,  epoch * global_steps + index)
      writer.add_scalar('current loss ', loss.item(), epoch * global_steps + index)

      index += 1
      global_steps += 1
      running_loss += loss
      
      # tepoch.set_postfix(loss = loss.item(), log_det=log_det.mean(0).item())  
      # sleep(0.001)
      # print(' Current learning rate (g:%d, p:%d): %.10f | Loss: %.6f'%(group_idx, param_idx, current_lr, loss.item()))
    
    mean_loss = sum(epoch_loss)/len(epoch_loss)
    writer.add_scalar('mean eocpch loss', mean_loss, epoch)
    print('epoch: ', epoch, 'epoch avg loss ', mean_loss.item())
    # scheduler.step(mean_loss)
    if epoch % 2 == 0: 
         torch.save({
             'epoch' : epoch,
             'model_state_dict' : glow.state_dict(),
             'optimizer_state_dict' : optimizer.state_dict(),
             'schedular_state_dict' : scheduler.state_dict(),
             'loss': epoch_loss
         },save_path    
     )

     

  if args.print_dict:
    print('Models state.dict:')
    for param_tensor in glow.state_dict():
        print(param_tensor, '\t', glow.state_dict()[param_tensor].size())
    print('Optimizers state_dict') 
    for var_name in optimizer.state_dict():
      print(var_name, '\t', optimizer.state_dict()[var_name])

if __name__ == '__main__':
    device = 'cpu'
    parser = argparse.ArgumentParser(description="Glow Model")
    parser.add_argument('--batch_size',default=16, type=int, help='size of the batch')
    parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='The learning rate for the optimizer')
    parser.add_argument('--actnorm', default=True, type=bool, help='Use activation normalization')
    parser.add_argument('--batchnorm', default=False, type=bool, help='Use batch normalization')
    parser.add_argument('--resNet', default=False, type=bool, help='Use ResNet normalization')
    parser.add_argument('--lu', default=True, type=bool, help='Use lu decompostion for the inverse convolution')
    parser.add_argument('--n_blocks', default=20, type=int, help='Number of blocks of the flow')
    parser.add_argument('--levels', default=2, type=int, help='Number of levels of flow')
    parser.add_argument('--n_bits', default=5, type=int, help='number of bits') 
    parser.add_argument('--n_samples', default=10, type=int, help=' Number of images sampled')
    parser.add_argument('--split', default=False, type=bool, help='Split the training data')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Choose a dataset, MNIST, FashionMNIST, CIFAR10')
    parser.add_argument('--affine', default=False, type=bool, help='Choose affine coupling if True or Addittive if False')
    parser.add_argument('--c', default=1, type=int, help='Number of channels')
    parser.add_argument('--print_dict', default=False, type=bool, help='Print model parameters')
    parser.add_argument('--dequantize', default=False, type=bool, help='True to dequantize the data')

    writer = SummaryWriter("runs/MNIST")
    
    train(device, parser, writer)
    writer.close()