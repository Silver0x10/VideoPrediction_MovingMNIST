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

dev = 'cuda:0'


class ConvLSTMCell(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Implementation from the original repo: https://github.com/ndrplz/ConvLSTM_pytorch

        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class EncoderDecoder(pl.LightningModule):
    def __init__(self, n_f, n_ch, k_sz):
        super(EncoderDecoder, self).__init__()
        self.padding = k_sz//2
        self.loss_fn = nn.MSELoss()

        self.enc1 = ConvLSTMCell(input_dim=n_f,
                                 hidden_dim=n_f,
                                 kernel_size=k_sz,
                                 bias=True)

        self.enc2 = ConvLSTMCell(input_dim=n_f,
                                 hidden_dim=n_f,
                                 kernel_size=k_sz,
                                 bias=True)

        self.dec1 = ConvLSTMCell(input_dim=n_f,
                                 hidden_dim=n_f,
                                 kernel_size=k_sz,
                                 bias=True)

        self.dec2 = ConvLSTMCell(input_dim=n_f,
                                 hidden_dim=n_f,
                                 kernel_size=k_sz,
                                 bias=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=n_ch,
                                             out_channels=n_f//2,
                                             kernel_size=k_sz,
                                             padding=self.padding),
                                   nn.BatchNorm2d(n_f//2),
                                   nn.GELU()
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=n_f//2,
                                             out_channels=n_f,
                                             kernel_size=k_sz,
                                             padding=self.padding),
                                   nn.BatchNorm2d(n_f),
                                   nn.GELU()
                                   )

        self.deconv1 = nn.Sequential(nn.Conv2d(in_channels=n_f,
                                               out_channels=n_f//2,
                                               kernel_size=k_sz,
                                               padding=self.padding),
                                     nn.BatchNorm2d(n_f//2),
                                     nn.GELU()
                                     )

        self.deconv2 = nn.Sequential(nn.Conv2d(in_channels=n_f//2,
                                               out_channels=n_ch,
                                               kernel_size=k_sz,
                                               padding=self.padding),
                                     nn.Tanh()
                                     )

    def autoencoder(self, x, frames, n_predictions, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        # Encoder
        for i in range(frames):

            x_conv_1 = self.conv1(x[:, i, :, :, :])  # 32
            x_conv_2 = self.conv2(x_conv_1)  # 64
            h_t1, c_t1 = self.enc1(input_tensor=x_conv_2,  # 32
                                   cur_state=[h_t1, c_t1]
                                   )

            h_t2, c_t2 = self.enc2(input_tensor=h_t1,  # 16
                                   cur_state=[h_t2, c_t2]
                                   )

        # Decoder
        for i in range(n_predictions):

            x_conv_1 = self.conv1(x[:, i, :, :, :])  # 32
            skip_A = x_conv_1

            x_conv_2 = self.conv2(x_conv_1)  # 64
            skip_B = x_conv_2

            h_t1, c_t1 = self.enc1(input_tensor=x_conv_2,  # 32
                                   cur_state=[h_t1, c_t1]
                                   )
            skip_C = h_t1

            h_t2, c_t2 = self.enc2(input_tensor=h_t1,  # 16
                                   cur_state=[h_t2, c_t2]
                                   )

            h_t3, c_t3 = self.dec1(input_tensor=h_t2,  # 32
                                   cur_state=[h_t3, c_t3]
                                   )
            # print('h_t3',h_t3[0,0,20:40,20:40])
            h_t4, c_t4 = self.dec2(input_tensor=h_t3,  # 64
                                   cur_state=[h_t4, c_t4]
                                   )
            # print('h_t4',h_t4[0,0,20:40,20:40])

            x_deconv_1 = self.deconv1(torch.add(h_t4, skip_B))  # 32
            # print('x_deconv_1',x_deconv_1[0,0,20:40,20:40])

            x_deconv_2 = self.deconv2(torch.add(x_deconv_1, skip_A))  # 1
            # print('x_deconv_2',x_deconv_2[0,0,20:40,20:40])

            out = x_deconv_2*10

            if i == 0:
                seq = out.unsqueeze(1)
                #print('SEQ SIZE =',seq.size())
            else:
                seq = torch.cat((seq, out.unsqueeze(1)), 2)

        seq = seq.squeeze()
        #print('SEQ PRIMA',seq[0,0,20:30,20:30])
        outputs = torch.nn.Sigmoid()(seq)*255.0
        #print('SEQ DOPO',outputs[0,0,20:30,20:30])
        return outputs

    def forward(self, x, n_p=10):
        """
        x must have [B,T,C,H,W] 
           B = Batch size
           T = Time, i.e n° of frames
           C = Channels
           H = Height
           W = Width
        """
        x = torch.unsqueeze(x, 2)  # Add channel size, that is equal to 1
        x.to(device=dev)
        b_s, frs, _, h, w = x.size()

        # inizialization of the hidden states
        h_t1, c_t1 = self.enc1.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t2, c_t2 = self.enc2.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t3, c_t3 = self.dec1.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t4, c_t4 = self.dec2.init_hidden(batch_size=b_s, image_size=(h, w))
        # nn.init.orthogonal_(self.conv_3D.weight)

        outputs = self.autoencoder(
            x, frs, n_p, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        out = self(x)

        loss = self.loss(y, out)
        # print('LOSS',loss)
        self.log("mse", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        out = self(x)

        loss = self.loss(y, out)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        out = self(x)

        loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        # optimizer = optim.RMSprop(self.parameters(), lr=0.01)#, weight_decay=1e-5)
        return optimizer

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
