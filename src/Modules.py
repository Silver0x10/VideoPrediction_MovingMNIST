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

from parameters import B,T,C,H,W


class Conv2D(nn.Module):

    def __init__(self, Size_in, Size_out, kernel_size, stride=1, padding=0,):
        super(Conv2D,self).__init__()
        self.conv = nn.Conv2d(Size_in, Size_out, kernel_size, stride, padding)
        #self.norm = nn.BatchNorm2d()
        self.activ = nn.ReLU()
        
    def forward(self,x):
        y = self.conv(x)
        return y
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dec = None



    