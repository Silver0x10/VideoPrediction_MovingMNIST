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

class PlEncoderDecoder(pl.LightningModule):
    def __init__(self, k_s, Batch_size, C=1):
        super(PlEncoderDecoder,self).__init__()
        d_p = (k_s - 1) // 2

        self.loss_fn = nn.MSELoss(reduction='mean')
        self.Conv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding = d_p),
                                  nn.ReLU(),
                                  #nn.BatchNorm2d(C)  
                                 )
        self.Conv_dwsamp = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding=d_p),
                                         nn.ReLU(),
                                         #nn.BatchNorm2d(C)
                                        )
        self.Deconv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding=d_p),
                                    nn.ReLU(),
                                    #nn.BatchNorm2d(C)
                                   )
        self.Deconv_upsamp = nn.Sequential(nn.Conv2d(C, C*4, kernel_size=k_s,padding=d_p),
                                          nn.PixelShuffle(upscale_factor= 2),
                                          nn.ReLU(),
                                          #nn.BatchNorm2d(C)
                                          )
        self.BN = nn.BatchNorm2d(C)
        self.lstm = nn.LSTM(256, 256, batch_first = True, dropout =0.1)
        self.B_s = Batch_size


    def forward(self, x, h = None):
        #print(' x  size = ',x .size())
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            #print('Start. x_frame size = ',x_frame.size())
            z = self.Conv(x_frame)
            #print('A',z.size())
            z_skip = self.Conv(z)
            #print('B',z.size())
            z = self.Conv_dwsamp(z_skip)
            #print('C',z.size())
            z = self.Conv(z)
            #print('D',z.size())
            z = self.Conv_dwsamp(z)
            #print('Convs end. z size = ',z.size())
            end_conv_size = z.size()[-1]
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            #print('Float size of z is =',z.size())
            lstm_out, h = self.lstm(z, h)
            #print('LSTM end. h size = ',lstm_out.size())
            lstm_out =lstm_out.view(-1,1,end_conv_size,end_conv_size) #Applied 2 times because Decoder need [B,C,W,H] shape
            lstm_out = self.BN(lstm_out)
            #DECODER
            #print('After reshaping, LSTM output size =', lstm_out.size())
            z = self.Deconv(lstm_out)
            #print( 'Deconv 1. z isze = ',z.size())
            z= self.Deconv_upsamp(z)
            #print( 'Deconv 2. z isze = ',z.size())
            z = self.Deconv(z)
            #print( 'Deconv 3. z isze = ',z.size())
            z= self.Deconv_upsamp(z)            
            #print('Deconv end. z size = ',z.size())
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)

        # From here on we predict the last 10 frames
        out = None
        prev_frame = last_real.unsqueeze(1)
        for i in range(0,x.size(1)):
            x_frame = prev_frame.squeeze(1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            z = self.Conv_dwsamp(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(-1,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
            lstm_out = self.BN(lstm_out)
            #DECODER
            z = self.Deconv(lstm_out)
            z= self.Deconv_upsamp(z)
            z = self.Deconv(z)
            z= self.Deconv_upsamp(z)
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
            prev_frame = out_frame
        return (out.squeeze(), h)
    

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        out, _ = self(x)

        loss = self.loss(out, y)
        self.log("mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        out, _ = self(x)

        loss = self.loss(out, y)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        out, _ = self(x)

        loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = optim.RMSprop(self.parameters(), lr=0.01)#, weight_decay=1e-5)
        return optimizer
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)