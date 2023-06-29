from typing import Any
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
import torch
from torch import optim, FloatTensor
import torch.nn as nn
import io
import imageio
from ipywidgets import widgets, HBox
from IPython.display import display
import matplotlib as plttorch

from src.parameters import B,T,C,H,W,mid_size,out_size


class Conv2D(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super(Conv2D,self).__init__()
        self.padding = 1#(kernel_size - stride) // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, self.padding)
        #self.norm = nn.BatchNorm2d()
        self.activ = nn.ReLU()
        
    def forward(self,x):
        y = self.conv(x)
        y = self.activ(y)
        #print('forward encoder check AAA, output size = ',y.size())

        return y
    
class ConvTranspose2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, H=0, W=0, output_padding=0, stride=1, upsampling = False,) -> None:
        super(ConvTranspose2D,self).__init__()
        padding = (kernel_size - stride) // 2
        o_p = output_padding
        if upsampling==True:
             o_p = (64,64)#(H//2,W//2)
             self.convTran = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding,o_p)

        if upsampling==False:
            self.convTran = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding, o_p)

        #self.norm = nn.BatchNorm2d()
        self.activ = nn.ReLU()

    def forward(self,x):
        y = self.convTran(x.float())
        y = self.activ(y)
        #print('forward decoder check BBB, output size = ',y.size())
        return y


    
class Encoder(nn.Module):
    """
    Following the implementation described in the paper, Encoder consists of 
    four vanilla 2D convolutional layers..unsqueeze() The first 2 have the same dim of the
    input, while the last 2 are smaller and have the same dimension
    """
    def __init__(self, C, k_s):
        super(Encoder,self).__init__()
        #Layers with stride=2 work as downsampling layers
        #The number of channels remains the same in all the layers
        self.encoder = nn.Sequential(Conv2D(C, C, kernel_size=k_s),
                                 Conv2D(C, C, kernel_size=k_s, stride=2),
                                 Conv2D(C, C, kernel_size=k_s),
                                 Conv2D(C, C, kernel_size=k_s, stride=2)
                                 )
        
    def forward(self,x):
        y = self.encoder(x.float()).float()
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
        self.decoder = nn.Sequential(Conv2D(C, C, kernel_size=kernel_size),
                                 Conv2D(C, C, kernel_size=kernel_size),
                                 nn.Upsample(scale_factor=2),
                                 Conv2D(C, C, kernel_size=kernel_size),
                                 Conv2D(C, C, kernel_size=kernel_size),
                                 nn.Upsample(scale_factor=2)
                                 )
        
    def forward(self,x):
        y = self.decoder(x).float()
        return y
        
        
class PlEncoderDecoder(pl.LightningModule):
    def __init__(self,encoder,decoder):
        super(PlEncoderDecoder,self).__init__()
        self.lstm = nn.LSTM(256, 256)
        #self.linear = nn.Linear()    
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y']
        

        y = y.float()
        #print('Y size = ',y.size())
        #print("X size = ",x.size())

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            #print("X size = ",x.size())
            #B,T,H,W = x.size()
            x_frame = x[:,i,:,:].float()
            #print("Frame size = ",x_frame.size())
            z = self.encoder(x_frame)
            #print('ENCODER FINITO')
            #print('z shape =',z.size())
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out = torch.unsqueeze(torch.unsqueeze(lstm_out.view(16,16),0),0) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print("Dopo la modifica, lstm_out = ", lstm_out.size())
            out= self.decoder(lstm_out)
            #print('DECODER FINITO')
            #print('out size = ',out.size())
        loss = nn.functional.mse_loss(out, y)
        #print('LOSS = ',loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
# encoder = Encoder(C,3)
# decoder = Decoder(C,3)
# autoencoder = PlEncoderDecoder(encoder,decoder)
