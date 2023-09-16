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

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
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
        super(EncoderDecoder,self).__init__()
        self.padding =  k_sz//2
        self.loss_fn = nn.MSELoss()

        self.enc1 = ConvLSTMCell(input_dim=n_ch*5,
                                 hidden_dim=n_f//2,
                                 kernel_size=k_sz,
                                 bias=True)

        self.enc2 = ConvLSTMCell(input_dim=n_f//2,
                                 hidden_dim=n_f,
                                 kernel_size=k_sz,
                                 bias=True) 

        self.dec1 = ConvLSTMCell(input_dim=n_f,
                                 hidden_dim=n_f//2,
                                 kernel_size=k_sz,
                                 bias=True) 

        self.dec2 = ConvLSTMCell(input_dim=n_f//2,
                                 hidden_dim=n_ch*5,
                                 kernel_size=k_sz,
                                 bias=True)

        self.conv = nn.Conv2d(in_channels=n_ch,
                                 out_channels=n_ch*5,
                                 kernel_size=k_sz,
                                 padding=self.padding
                                 )
        self.deconv = nn.Conv2d(in_channels = n_ch*5,
                                out_channels= n_ch,
                                kernel_size=k_sz,
                                padding=self.padding)
        
    def autoencoder(self, x, frames, n_predictions, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        #Encoder  
        for i in range(frames):
            
            x_conved = self.conv(x[:,i,:,:,:])
            h_t1,c_t1 = self.enc1(input_tensor =  x_conved,
                                  cur_state = [h_t1,c_t1]
                                  )
            h_t2,c_t2 = self.enc2(input_tensor = h_t1,
                                  cur_state = [h_t2,c_t2]
                                  )
        encoder_output = h_t2
        skip_layer = x_conved

        #Decoder
        for i in range(n_predictions):

            h_t3,c_t3 = self.dec1(input_tensor = encoder_output,
                                  cur_state = [h_t3,c_t3]
                                  )
                
            h_t4,c_t4 = self.dec2(input_tensor =  h_t3,
                                  cur_state = [h_t4,c_t4]
                                  )
            #Here i add back the skip layer
            last_layer = self.deconv(torch.add(skip_layer,h_t4)) #Should be of dim [b_s, 5, w, h]
            decoder_output=last_layer
            #Create a new skip layer
            skip_layer = self.conv(last_layer)
            if i == 0:
                seq = decoder_output.unsqueeze(1)
                #print('SEQ SIZE =',seq.size())
            else:
                seq = torch.cat((seq, decoder_output.unsqueeze(1)), 2)
                #print('SEQ SIZE =',seq.size())
        #seq dovrebbe avere dim = [b_s, n_features, n_p, h, w] ([16,64,10,64,64])
        #Lui fa conv3D invertendo in seconda posizione n_f(=64) e n_p(=10). Provo a fare così ma non mi piace.
        #In caso cambio.
        #outputs = self.conv_3D(seq)
        #print('CIAO SONO GIACOMO',seq[0,0,5,20:40,20:40])
        #print('MAX of pre sifmoid',torch.max(seq[0,0,5,:,:]))



        # print('studio del output.Vediamo se ci sono valori maggiori di 1')
        # for i in range(16):
        #     indici_positivi = (seq[i,0,0,:,:] > 1).nonzero()

        #     # Stampa i valori maggiori di zero
        #     for indice in indici_positivi:
        #         riga, colonna = indice.tolist()
        #         valore = seq[i,0,0,riga, colonna].item()
        #         print(f"Valore positivo trovato: {valore} alla posizione [riga {riga}, colonna {colonna}]")


        outputs = torch.nn.Sigmoid()(seq)
        #print('E ORA SONO COSI',outputs[0,0,5,20:40,20:40])
        #print('MAX of output',torch.max(outputs[0,0,5,:,:]))
        return outputs
    
    def forward(self, x, n_p = 10):       
        """
        x must have [B,T,C,H,W] 
           B = Batch size
           T = Time, i.e n° of frames
           C = Channels
           H = Height
           W = Width
        """
        x = torch.unsqueeze(x,2) #Add channel size, that is equal to 1
        b_s, frs, _, h, w = x.size()

        # inizialization of the hidden states
        h_t1, c_t1 = self.enc1.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t2, c_t2 = self.enc2.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t3, c_t3 = self.dec1.init_hidden(batch_size=b_s, image_size=(h, w))
        h_t4, c_t4 = self.dec2.init_hidden(batch_size=b_s, image_size=(h, w))
        #nn.init.orthogonal_(self.conv_3D.weight)

        outputs = self.autoencoder(x, frs, n_p, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        return outputs
    

    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        out= self(x).squeeze()*255.0
        # print('TRAINING: type dei frame x', x.dtype)
        # print('TRAINING: type dei groundtruth y', y.dtype)
        # print('TRAINING: type degli out', out.dtype)        

        # for i in range(64):
        #     for j in range(64):
        #         if y[0,0,i,j]!= 0. :

        #             print('CARRAMBA',y[0,0,i,j])
    

        #print('LA DIFF. TRA I DUE E',torch.sub(out,y)[0,0,20:40,20:40])
        loss = self.loss(y,out)
        #print('LOSS',loss)
        self.log("mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        out= self(x).squeeze()*255.0

        loss = self.loss(y,out)
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        x = x/255.0

        out = self(x).squeeze()*255.0

        loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        #optimizer = optim.RMSprop(self.parameters(), lr=0.01)#, weight_decay=1e-5)
        return optimizer
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)


            
        









