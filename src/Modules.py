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
        self.Conv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding = 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(C)  
                                 )
        self.Conv_dwsamp = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(C)
                                        )
        self.Deconv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(C)
                                   )
        self.Deconv_upsamp = nn.Sequential(nn.Conv2d(C, C*4, kernel_size=k_s,padding=1),
                                          nn.PixelShuffle(upscale_factor= 2),
                                          nn.ReLU(),
                                          nn.BatchNorm2d(C)
                                          )
        self.lstm = nn.LSTM(1024, 1024)
        self.B_s = Batch_size


    def forward(self, frame, h = None):
        frame = frame.float().unsqueeze(0)

        z = self.Conv(frame.unsqueeze(0))
        z_skip = self.Conv(z)
        z = self.Conv_dwsamp(z_skip)
        z = self.Conv(z)

        z = z.view(z.size(0),-1).float()
        lstm_out, h = self.lstm(z, h)
        lstm_out =lstm_out.view(1,1,32,32)

        z = self.Deconv(lstm_out)
        z = self.Deconv(z)
        z= self.Deconv_upsamp(z)
        z_reshape = self.Deconv(torch.add(z, z_skip))
        out = self.Deconv(z_reshape)

        return (out, h)
    

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape

        #DECODER
        z = self.Deconv(lstm_out)
        z = self.Deconv(z)
        z= self.Deconv_upsamp(z)
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)

        # From here on we predict the last 10 frames
        out = None
        for i in range(0,y.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape
            #DECODER
            z = self.Deconv(lstm_out)
            z = self.Deconv(z)
            z= self.Deconv_upsamp(z)
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape

        #DECODER
        z = self.Deconv(lstm_out)
        z = self.Deconv(z)
        z= self.Deconv_upsamp(z)
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)

        # From here on we predict the last 10 frames
        out = None
        for i in range(0,y.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape
            #DECODER
            z = self.Deconv(lstm_out)
            z = self.Deconv(z)
            z= self.Deconv_upsamp(z)
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
        loss = nn.functional.mse_loss(out, y)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape

        #DECODER
        z = self.Deconv(lstm_out)
        z = self.Deconv(z)
        z= self.Deconv_upsamp(z)
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)

        # From here on we predict the last 10 frames
        out = None
        for i in range(0,y.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #ENCODER
            z = self.Conv(x_frame)
            z_skip = self.Conv(z)
            z = self.Conv_dwsamp(z_skip)
            z = self.Conv(z)
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,32,32) #Applied 2 times because Decoder need [B,C,W,H] shape
            #DECODER
            z = self.Deconv(lstm_out)
            z = self.Deconv(z)
            z= self.Deconv_upsamp(z)
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
        loss = nn.functional.mse_loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    
# encoder = Encoder(C,3)
# decoder = Decoder(C,3)
# autoencoder = PlEncoderDecoder(encoder,decoder)


#from src.parameters import B,T,C,H,W,mid_size,out_size


# class Conv2D(nn.Module):

#     def __init__(self, C_in, C_out, kernel_size, stride=1):
#         super(Conv2D,self).__init__()
#         self.padding = 1#(kernel_size - stride) // 2
#         self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, self.padding)
#         #self.norm = nn.BatchNorm2d()
#         self.activ = nn.ReLU()
        
#     def forward(self,x):
#         y = self.conv(x)
#         y = self.activ(y)
#         #print('forward encoder check AAA, output size = ',y.size())

#         return y
    
# class ConvTranspose2D(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, H=0, W=0, output_padding=0, stride=1, upsampling = False,) -> None:
#         super(ConvTranspose2D,self).__init__()
#         padding = (kernel_size - stride) // 2
#         o_p = output_padding
#         if upsampling==True:
#              o_p = (64,64)#(H//2,W//2)
#              self.convTran = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding,o_p)

#         if upsampling==False:
#             self.convTran = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding, o_p)

#         #self.norm = nn.BatchNorm2d()
#         self.activ = nn.ReLU()

#     def forward(self,x):
#         y = self.convTran(x.float())
#         y = self.activ(y)
#         #print('forward decoder check BBB, output size = ',y.size())
#         return y


    
# class Encoder(nn.Module):
#     """
#     Following the implementation described in the paper, Encoder consists of 
#     four vanilla 2D convolutional layers..unsqueeze() The first 2 have the same dim of the
#     input, while the last 2 are smaller and have the same dimension
#     """
#     def __init__(self, C, k_s):
#         super(Encoder,self).__init__()
#         #Layers with stride=2 work as downsampling layers
#         #The number of channels remains the same in all the layers
#         self.encoder = nn.Sequential(Conv2D(C, C, kernel_size=k_s),
#                                     Conv2D(C, C, kernel_size=k_s),
#                                     Conv2D(C, C, kernel_size=k_s, stride=2),
#                                     Conv2D(C, C, kernel_size=k_s)
#                                     )
        
#     def forward(self,x):
#         y = self.encoder(x.float()).float()
#         return y


# class Decoder(nn.Module):
#     """
#     Following the implementation described in the paper, Encoder consists of 
#     four vanilla 2D convolutional layers. The first 2 have the same dim of the
#     input, while the last 2 are smaller and have the same dimension
#     """
#     def __init__(self, C, kernel_size):
#         super(Decoder,self).__init__()
#         #L
#         #The number of channels remains the same in all the layers
#         self.decoder = nn.Sequential(Conv2D(C, C, kernel_size=kernel_size),
#                                  Conv2D(C, C*4, kernel_size=kernel_size),
#                                  nn.PixelShuffle(upscale_factor= 2),
#                                  Conv2D(C, C, kernel_size=kernel_size),
#                                  Conv2D(C, C, kernel_size=kernel_size),
#                                  nn.Upsample(scale_factor=2)
#                                  )
        
#     def forward(self,x):
#         y = self.decoder(x).float()
#         return y