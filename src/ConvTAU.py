
from typing import Any
import torch
from torch import optim, nn
import lightning.pytorch as pl

from src.parameters import ParamsConvTAU

class TAU(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        k_s = params.kernel_size
        dim = params.frames_per_sample
        dilation = params.dilation 
        
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = k_s // dilation + ((k_s // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        
        # Statical Attention Modules
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=d_k, padding=d_p, groups=dim)
        self.dw_d_conv = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.onebyone_conv = nn.Conv2d(dim, dim, 1)
        
        # Dynamical Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, params.fc_hidden_dim, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(params.fc_hidden_dim, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    
    def forward(self, x):
        skip = x.clone()

        # Statical Attention
        sa = self.onebyone_conv( self.dw_d_conv( self.dw_conv(x) ) )
        
        # Dynamical Attention
        da = self.avgpool(x).view(x.shape[0], x.shape[1])
        da = self.fc(da).view(x.shape[0], x.shape[1], 1, 1)
        
        return sa * da * skip
    
        
class Encoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        C = params.channels
        k_s = params.kernel_size
        
        self.encoder = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU()
                                        )
        
    def forward(self, x):
        skip = None
        encoded_frames = x
        
        for i in range(len(self.encoder)):
            encoded_frames = self.encoder[i](encoded_frames)
            if i==3: skip = encoded_frames
            
        return encoded_frames, skip


class Decoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        C = params.channels
        k_s = params.kernel_size
        
        self.decoder = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C*4, kernel_size=k_s, padding = 1),
                                        nn.PixelShuffle(upscale_factor= 2),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU()
                                        )
        # self.readout = nn.Conv2d(19, C, 1)
        
    def forward(self, x, skip):
        decoded_frames = x

        for i in range(len(self.decoder)):
            # if i==7: decoded_frames = torch.add(decoded_frames, skip)
            if i==7: decoded_frames += skip
            decoded_frames = self.decoder[i](decoded_frames)
        
        # decoded_frames = self.readout(decoded_frames)
        
        return decoded_frames
    

class ConvTAU(pl.LightningModule):
    def __init__(self, params):
        super().__init__()        
        C = params.channels
        k_s = params.kernel_size
        
        self.loss_fn = nn.MSELoss();

        self.encoder = Encoder(params)
        self.tau = TAU(params)
        self.decoder = Decoder(params)
        
    
    def forward(self, frames):
        frames = frames.unsqueeze(0)
        B, T, H, W = frames.shape
        frames = frames.view(B*T, 1, H, W)
        
        h, skip = self.encoder(frames)
        BT, C_, H_, W_ = h.shape
        
        h = h.view(B, T*C_, H_, W_)
        tau_out = self.tau(h)
        tau_out = tau_out.view(B*T, C_, H_, W_)
        
        out = self.decoder(tau_out, skip)
        out = out.view(B, T, 1, H, W)
        
        return out.squeeze(0)
    
        
    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        
        h, skip = self.encoder(x)
        BT, C_, H_, W_ = h.shape
        
        h = h.view(B, T*C_, H_, W_)
        tau_out = self.tau(h)
        tau_out = tau_out.view(B*T, C_, H_, W_)
        
        out = self.decoder(tau_out, skip)
        out = out.view(B, T, 1, H, W)
        print(out.shape)

        loss = self.loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss(self, pred, y):
        # TODO KULLBACK-LEIBLER divergence
        return self.loss_fn(pred, y)
        
