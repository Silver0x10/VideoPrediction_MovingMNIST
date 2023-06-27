import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import io
import imageio
from ipywidgets import widgets, HBox
from IPython.display import display
import matplotlib as plt

from parameters import B,T,C,H,W,mid_size,out_size


class Conv2D(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super(Conv2D,self).__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
        #self.norm = nn.BatchNorm2d()
        self.activ = nn.ReLU()
        
    def forward(self,x):
        y = self.conv(x)
        y = self.activ(y)
        return y
    
class ConvTranspose2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, output_padding=0, stride=1, upsampling = False) -> None:
        super(ConvTranspose2D,self).__init__()
        padding = (kernel_size - stride) // 2
        self.upsampling = upsampling
        self.convTran = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding, output_padding, upsampling)
        #self.norm = nn.BatchNorm2d()
        self.activ = nn.ReLU()

    def forward(self,x):
        C,H,W = x.shape()
        if self.upsampling== True:
            # We double the size of the output since in Encoder we halfed them
            output_padding = (H//2, W//2)
            y = self.convTran(x,output_padding=output_padding)
        else:
            y = self.convTran(x)
        y = self.activ(y)
        return y

        

    
class Encoder(nn.Module):
    """
    Following the implementation described in the paper, Encoder consists of 
    four vanilla 2D convolutional layers. The first 2 have the same dim of the
    input, while the last 2 are smaller and have the same dimension
    """
    def __init__(self, C, kernel_size):
        super(Encoder,self).__init__()
        #Layers with stride=2 work as downsampling layers
        #The number of channels remains the same in all the layers
        self.encoder = nn.Sequential(Conv2D(C, C, kernel_size=kernel_size),
                                 Conv2D(C, C, kernel_size=kernel_size, stride=2),
                                 Conv2D(C, C, kernel_size=kernel_size),
                                 Conv2D(C, C, kernel_size=kernel_size, stride=2)
                                 )
        
    def forward(self,x):
        # T,C,H,W = x.shape()
        for i in range[0,self.encoder]:
            y = self.encoder[i](x)
        return y


class Decoder(nn.Module):
    """
    Following the implementation described in the paper, Encoder consists of 
    four vanilla 2D convolutional layers. The first 2 have the same dim of the
    input, while the last 2 are smaller and have the same dimension
    """
    def __init__(self, C, kernel_size):
        super(Decoder,self).__init__()
        #L
        #The number of channels remains the same in all the layers
        self.decoder = nn.Sequential(ConvTranspose2D(C, C, kernel_size=kernel_size),
                                 ConvTranspose2D(C, C, kernel_size=kernel_size, upsampling=True),
                                 ConvTranspose2D(C, C, kernel_size=kernel_size),
                                 ConvTranspose2D(C, C, kernel_size=kernel_size, upsampling=True)
                                 )
        
    def forward(self,x):
        T,C,H,W = x.shape()
        for i in range[0,self.decoder]:
            y = self.decoder[i](x)
        return y
        
        
        



    