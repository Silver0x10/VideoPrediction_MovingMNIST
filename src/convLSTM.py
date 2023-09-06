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
        self.Conv_L = nn.Sequential(nn.Conv2d(C, 4*C, kernel_size=k_s,padding = 3),
                                  nn.ReLU(),
                                  #nn.BatchNorm2d(4*C)  
                                 )
        self.Conv_XL = nn.Sequential(nn.Conv2d(4*C, 4*C, kernel_size=k_s,padding = 3),
                            nn.ReLU(),
                            #nn.BatchNorm2d(4*C)  
                            )
        self.Conv_S = nn.Sequential(nn.Conv2d(4*C, C, kernel_size=k_s,padding = 2),
                    nn.ReLU(),
                    #nn.BatchNorm2d(4*C)  
                    )
        self.Conv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding = 2),
                                  nn.ReLU(),
                                  #nn.BatchNorm2d(C)  
                                 )        
        self.Conv_dwsamp = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding=3),
                                         nn.ReLU(),
                                         #nn.BatchNorm2d(C)
                                        )
        self.pooling_L = nn.AdaptiveAvgPool2d((32,32))
        self.pooling_XL = nn.AdaptiveAvgPool2d((16,16))
        self.Deconv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, padding=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(C)
                                   )
        self.Deconv_upsamp = nn.Sequential(nn.Conv2d(C, C*4, kernel_size=k_s,padding=3),
                                          nn.PixelShuffle(upscale_factor= 2),
                                          nn.ReLU(),
                                          #nn.BatchNorm2d(C)
                                          )
        self.DecT= nn.ConvTranspose2d(C, C, kernel_size=6, padding=2, stride=2)
        self.lstm = nn.LSTM(256, 64, batch_first = True,dropout = 0.25)
        self.B_s = Batch_size


    def forward(self, frames, h = None):
        #h = None
        lstm_out = None
        x = frames
        for i in range(0,x.size(0)):
            x_frame = torch.unsqueeze(x[i,:,:].float(),0)
            print('X-frame size =',x_frame.type())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(1,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)
        #print('Last real =', last_real.size())

        # From here on we predict the last 10 frames
        out = None
        prev_frame = last_real
        print('PREV FRAME SIZE=',prev_frame.size())
        for i in range(0,x.size(0)):
            x_frame = prev_frame.squeeze(1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            #print('E qua quanto è z_skip?',z_skip.size())
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(1,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
            prev_frame = out_frame

        return out
    

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)
        #print('Last real =', last_real.size())

        # From here on we predict the last 10 frames
        out = None
        prev_frame = last_real.unsqueeze(1)
        for i in range(0,y.size(1)):
            x_frame = prev_frame.squeeze(1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            #print('E qua quanto è z_skip?',z_skip.size())
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
            prev_frame = out_frame

        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)
        #print('Last real =', last_real.size())

        # From here on we predict the last 10 frames
        out = None
        prev_frame = last_real.unsqueeze(1)
        for i in range(0,y.size(1)):
            x_frame = prev_frame.squeeze(1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            #print('E qua quanto è z_skip?',z_skip.size())
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
            prev_frame = out_frame

        loss = nn.functional.mse_loss(out, y)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

        h = None
        lstm_out = None
        for i in range(0,x.size(1)):
            x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
        z_reshape = self.Deconv(torch.add(z, z_skip))
        last_real = self.Deconv(z_reshape)
        #print('Last real =', last_real.size())

        # From here on we predict the last 10 frames
        out = None
        prev_frame = last_real.unsqueeze(1)
        for i in range(0,y.size(1)):
            x_frame = prev_frame.squeeze(1)
            #print('X-frame size =',x_frame.size())
            #ENCODER
            z = self.Conv(x_frame)
            #print('Prima convolution size =',z.size())
            z_skip = self.Conv(z)
            #print('E qua quanto è z_skip?',z_skip.size())
            z = self.Conv_L(z)
            z = self.pooling_L(z)
            #print('PUPPY',z.size())
            z = self.Conv_XL(z)
            #print('A',z.size())
            z = self.Conv_S(z)
            #print('B',z.size())
            z = self.pooling_XL(z)
            #print('C',z.size())
            #LATENT SPACE
            z = z.view(z.size(0),-1).float()
            lstm_out, h = self.lstm(z, h)
            #print("lstm_out", lstm_out.size())
            lstm_out =lstm_out.view(self.B_s,1,8,8) #Applied 2 times because Decoder need [B,C,W,H] shape
            #print('lstm_out',lstm_out.size())
            #DECODER
            z = self.DecT(lstm_out)
            #print('D',z.size())
            z = self.Deconv(z)
            #print('E',z.size())
            z= self.DecT(z)
            #print('F',z.size())
            z = self.Deconv(z)
            #print('G',z.size())
            z= self.DecT(z)
            #print('M',z.size())
            z_reshape = self.Deconv(torch.add(z, z_skip))
            out_frame = self.Deconv(z_reshape).unsqueeze(1)
            if i == 0:
                out = out_frame
            else:
                out = torch.cat((out,out_frame),1)
            prev_frame = out_frame
        loss = nn.functional.mse_loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4,weight_decay=1e-5)
        return optimizer



###################################################################################################################################################


# class PlEncoderDecoder(pl.LightningModule):
#     def __init__(self, k_s, Batch_size, C=1):
#         super(PlEncoderDecoder,self).__init__()
#         self.Conv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding = 1),
#                                   nn.ReLU(),
#                                   #nn.BatchNorm2d(C)  
#                                  )
#         self.Conv_dwsamp = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding=1),
#                                          nn.ReLU(),
#                                          #nn.BatchNorm2d(C)
#                                         )
#         self.Deconv = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s,padding=3),
#                                     nn.ReLU(),
#                                     #nn.BatchNorm2d(C)
#                                    )
#         self.Deconv_upsamp = nn.Sequential(nn.Conv2d(C, C*4, kernel_size=k_s,padding=3),
#                                           nn.PixelShuffle(upscale_factor= 2),
#                                           nn.ReLU(),
#                                           #nn.BatchNorm2d(C)
#                                           )
#         self.lstm = nn.LSTM(81, 81, batch_first = True)
#         self.B_s = Batch_size


#     def forward(self, frame, h = None):
#         frame = frame.float().unsqueeze(0)

#         z = self.Conv(frame.unsqueeze(0))
#         z_skip = self.Conv(z)
#         z = self.Conv_dwsamp(z_skip)
#         z = self.Conv(z)
#         z = self.Conv_dwsamp(z)
#         z = z.view(z.size(0),-1).float()
#         lstm_out, h = self.lstm(z, h)
#         lstm_out =lstm_out.view(1,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#         z = self.Deconv(lstm_out)
#         z= self.Deconv_upsamp(z)
#         z = self.Deconv(z)
#         z= self.Deconv_upsamp(z)
#         z_reshape = self.Deconv(torch.add(z, z_skip))
#         out = self.Deconv(z_reshape)

#         return (out, h)
    

#     def training_step(self, batch, batch_idx):
#         x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
#         print(' x  size = ',x .size())
#         h = None
#         lstm_out = None
#         for i in range(0,x.size(1)):
#             x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
#             #ENCODER
#             print('Start. x_frame size = ',x_frame.size())
#             z = self.Conv(x_frame)
#             z_skip = self.Conv(z)
#             z = self.Conv_dwsamp(z_skip)
#             z = self.Conv(z)
#             z = self.Conv_dwsamp(z)
#             print('Convs end. z size = ',z.size())
#             end_conv_size = z.size()[-1]
#             #LATENT SPACE
#             z = z.view(z.size(0),-1).float()
#             print('Float size of z is =',z.size())
#             lstm_out, h = self.lstm(z, h)
#             print('LSTM end. h size = ',lstm_out.size())
#             lstm_out =lstm_out.view(self.B_s,1,end_conv_size,end_conv_size) #Applied 2 times because Decoder need [B,C,W,H] shape
#             #DECODER
#             print('After reshaping, LSTM output size =', lstm_out.size())
#             z = self.Deconv(lstm_out)
#             print( 'Deconv 1. z isze = ',z.size())
#             z= self.Deconv_upsamp(z)
#             print( 'Deconv 2. z isze = ',z.size())
#             z = self.Deconv(z)
#             print( 'Deconv 3. z isze = ',z.size())
#             z= self.Deconv_upsamp(z)            
#             print('Deconv end. z size = ',z.size())
#         z_reshape = self.Deconv(torch.add(z, z_skip))
#         last_real = self.Deconv(z_reshape)

#         # From here on we predict the last 10 frames
#         out = None
#         prev_frame = last_real.unsqueeze(1)
#         for i in range(0,y.size(1)):
#             x_frame = prev_frame.squeeze(1)
#             #ENCODER
#             z = self.Conv(x_frame)
#             z_skip = self.Conv(z)
#             z = self.Conv_dwsamp(z_skip)
#             z = self.Conv(z)
#             z = self.Conv_dwsamp(z)
#             #LATENT SPACE
#             z = z.view(z.size(0),-1).float()
#             lstm_out, h = self.lstm(z, h)
#             #print("lstm_out", lstm_out.size())
#             lstm_out =lstm_out.view(self.B_s,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#             #DECODER
#             z = self.Deconv(lstm_out)
#             z= self.Deconv_upsamp(z)
#             z = self.Deconv(z)
#             z= self.Deconv_upsamp(z)
#             z_reshape = self.Deconv(torch.add(z, z_skip))
#             out_frame = self.Deconv(z_reshape).unsqueeze(1)
#             if i == 0:
#                 out = out_frame
#             else:
#                 out = torch.cat((out,out_frame),1)
#             prev_frame = out_frame

#         loss = nn.functional.mse_loss(out, y)
#         self.log("train_loss", loss, on_epoch=True)
#         return loss
    

#     # def validation_step(self, batch, batch_idx):
#     #     x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

#     #     h = None
#     #     lstm_out = None
#     #     for i in range(0,x.size(1)):
#     #         x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
#     #         #ENCODER
#     #         z = self.Conv(x_frame)
#     #         z_skip = self.Conv(z)
#     #         z = self.Conv_dwsamp(z_skip)
#     #         z = self.Conv(z)
#     #         z = self.Conv_dwsamp(z)
#     #         #LATENT SPACE
#     #         z = z.view(z.size(0),-1).float()
#     #         lstm_out, h = self.lstm(z, h)
#     #         #print("lstm_out", lstm_out.size())
#     #         lstm_out =lstm_out.view(self.B_s,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#     #         #DECODER
#     #         z = self.Deconv(lstm_out)
#     #         z= self.Deconv_upsamp(z)
#     #         z = self.Deconv(z)
#     #         z= self.Deconv_upsamp(z)
#     #     z_reshape = self.Deconv(torch.add(z, z_skip))
#     #     last_real = self.Deconv(z_reshape)

#     #     # From here on we predict the last 10 frames
#     #     out = None
#     #     prev_frame = last_real.unsqueeze(1)
#     #     for i in range(0,y.size(1)):
#     #         x_frame = prev_frame.squeeze(1)
#     #         #ENCODER
#     #         z = self.Conv(x_frame)
#     #         z_skip = self.Conv(z)
#     #         z = self.Conv_dwsamp(z_skip)
#     #         z = self.Conv(z)
#     #         z = self.Conv_dwsamp(z)
#     #         #LATENT SPACE
#     #         z = z.view(z.size(0),-1).float()
#     #         lstm_out, h = self.lstm(z, h)
#     #         #print("lstm_out", lstm_out.size())
#     #         lstm_out =lstm_out.view(self.B_s,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#     #         #DECODER
#     #         z = self.Deconv(lstm_out)
#     #         z= self.Deconv_upsamp(z)
#     #         z = self.Deconv(z)
#     #         z= self.Deconv_upsamp(z)
#     #         z_reshape = self.Deconv(torch.add(z, z_skip))
#     #         out_frame = self.Deconv(z_reshape).unsqueeze(1)
#     #         if i == 0:
#     #             out = out_frame
#     #         else:
#     #             out = torch.cat((out,out_frame),1)
#     #         prev_frame = out_frame
#     #     loss = nn.functional.mse_loss(out, y)
#     #     self.log("validation_loss", loss, on_epoch=True)
#     #     return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)

#         h = None
#         lstm_out = None
#         for i in range(0,x.size(1)):
#             x_frame = torch.unsqueeze(x[:,i,:,:].float(),1)
#             #ENCODER
#             z = self.Conv(x_frame)
#             z_skip = self.Conv(z)
#             z = self.Conv_dwsamp(z_skip)
#             z = self.Conv(z)
#             z = self.Conv_dwsamp(z)
#             #LATENT SPACE
#             z = z.view(z.size(0),-1).float()
#             lstm_out, h = self.lstm(z, h)
#             #print("lstm_out", lstm_out.size())
#             lstm_out =lstm_out.view(self.B_s,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#             #DECODER
#             z = self.Deconv(lstm_out)
#             z= self.Deconv_upsamp(z)
#             z = self.Deconv(z)
#             z= self.Deconv_upsamp(z)
#         z_reshape = self.Deconv(torch.add(z, z_skip))
#         last_real = self.Deconv(z_reshape)

#         # From here on we predict the last 10 frames
#         out = None
#         prev_frame = last_real.unsqueeze(1)
#         for i in range(0,y.size(1)):
#             x_frame = prev_frame.squeeze(1)
#             #ENCODER
#             z = self.Conv(x_frame)
#             z_skip = self.Conv(z)
#             z = self.Conv_dwsamp(z_skip)
#             z = self.Conv(z)
#             z = self.Conv_dwsamp(z)
#             #LATENT SPACE
#             z = z.view(z.size(0),-1).float()
#             lstm_out, h = self.lstm(z, h)
#             #print("lstm_out", lstm_out.size())
#             lstm_out =lstm_out.view(self.B_s,1,16,16) #Applied 2 times because Decoder need [B,C,W,H] shape
#             #DECODER
#             z = self.Deconv(lstm_out)
#             z= self.Deconv_upsamp(z)
#             z = self.Deconv(z)
#             z= self.Deconv_upsamp(z)
#             z_reshape = self.Deconv(torch.add(z, z_skip))
#             out_frame = self.Deconv(z_reshape).unsqueeze(1)
#             if i == 0:
#                 out = out_frame
#             else:
#                 out = torch.cat((out,out_frame),1)
#             prev_frame = out_frame
#         loss = nn.functional.mse_loss(out, y)
#         self.log("test_loss", loss, on_epoch=True)
#         return loss


#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
#         return optimizer