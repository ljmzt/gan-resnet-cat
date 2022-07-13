# implement the Fig 8, Table 4 resnet in the SNGAN paper

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as sn

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, sample='same', 
                 use_bn=True, use_sn=False):
        super().__init__()
        if (sample == 'up'):
            self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        elif (sample == 'same'):
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        elif (sample == 'down'):
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        if (sample == 'up'):
            self.conv_shortcut = nn.ConvTranspose2d(in_channel, out_channel, 1, 2, 0, 1)
        elif (sample == 'down'):
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 2, 0)
        elif (sample == 'same'):
            self.conv_shortcut = nn.Identity()
        
        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
        else:
            self.bn1 = self.bn2 = nn.Identity()
            
        if use_sn:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if (sample != 'same'):  self.conv_shortcut = sn(self.conv_shortcut)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        shortcut = self.conv_shortcut(x)
        x = self.conv1(self.activation(self.bn1(x)))
        x = self.conv2(self.activation(self.bn2(x)))
        return x + shortcut

class Discriminator(nn.Module):
    def __init__(self, img_size, dim=128, use_bn=False, use_sn=False):
        ''' assume H=W and multiples of 4 '''
        super().__init__()
        C, H, W = img_size
        self.conv = nn.Sequential(Block(C, dim, 'down', use_bn, use_sn),
                                  Block(dim, dim, 'down', use_bn, use_sn),
                                  Block(dim, dim, 'same', use_bn, use_sn),
                                  Block(dim, dim, 'same', use_bn, use_sn))
        self.activation = nn.ReLU()
        self.pool = nn.AvgPool2d(H//4)
        self.fc = nn.Linear(dim, 1)
        if use_sn:  self.fc = sn(self.fc)
    
    def forward(self, x):
        ''' example C,H,W = 3*32*32 '''
        x = self.conv(x)   #3*32*32 -> 128*8*8
        x = self.pool(self.activation(x)).reshape(x.shape[0],-1) # 128
        x = self.fc(x)
        return x.reshape(-1)

class Generator(nn.Module):
    def __init__(self, img_size, dim=256, zdim=128, use_bn=True):
        ''' assume H=W and multiples of 8 '''
        super().__init__()
        C, H, W = img_size
        self.H = H
        self.W = W
        self.dim = dim
        self.fc = nn.Linear(zdim, dim*H//8*W//8)
        self.conv = nn.Sequential(Block(dim, dim, 'up', use_bn),
                                  Block(dim, dim, 'up', use_bn),
                                  Block(dim, dim, 'up', use_bn))
        self.extra = nn.Sequential(nn.BatchNorm2d(dim),
                                   nn.ReLU(),
                                   nn.Conv2d(dim, C, 3, 1, 1))
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.dim, self.H//8, self.W//8)
        x = self.conv(x)
        x = self.extra(x)
        return nn.Tanh()(x)

