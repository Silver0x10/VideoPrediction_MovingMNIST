
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
        
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        
        # Statical Attention Modules
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=d_k, padding=d_p, groups=dim)
        self.dw_d_conv = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.onebyone_conv = nn.Conv2d(dim, dim, 1)
        
        # Dynamical Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, params.fc_hidden_dim, bias=False), # reduction
            nn.ReLU(),
            nn.Linear(params.fc_hidden_dim, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    
    def forward(self, x):
        skip = x.clone()
        
        x = self.activation( self.proj_1(x) )

        # Statical Attention
        sa = self.onebyone_conv( self.dw_d_conv( self.dw_conv(x) ) )
        
        # Dynamical Attention
        da = self.avgpool(x).view(x.shape[0], x.shape[1])
        da = self.fc(da).view(x.shape[0], x.shape[1], 1, 1)
        
        out = sa * da * skip
        
        out = torch.add(self.proj_2(out), skip)
        
        return out
    
        
class Encoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        C = params.channels
        k_s = params.kernel_size
        
        self.encoder = nn.Sequential(nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        # nn.Conv2d(C, C, kernel_size=k_s, stride=2, padding = 1),
                                        nn.Conv2d(C, C, kernel_size=k_s, padding = 1),
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
        
        self.decoder = nn.Sequential(nn.ConvTranspose2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        # nn.Conv2d(C, C*4, kernel_size=k_s, padding = 1),
                                        # nn.PixelShuffle(upscale_factor= 2),
                                        # nn.ReLU(),
                                        nn.ConvTranspose2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(C, C, kernel_size=k_s, padding = 1),
                                        nn.ReLU()
                                        )
        # self.readout = nn.Conv2d(19, C, 1)
        
    def forward(self, x, skip):
        decoded_frames = x

        for i in range(len(self.decoder)):
            if i==7: decoded_frames = torch.add(decoded_frames, skip)
            decoded_frames = self.decoder[i](decoded_frames)
        
        # decoded_frames = self.readout(decoded_frames)
        
        return decoded_frames
    

def kullback_leibler_divergence(pred_y, batch_y, tau=0.1, eps=1e-12):
    B, T, C = pred_y.shape[:3]
    if T <= 2:  return 0
    gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
    gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
    softmax_gap_p = nn.functional.softmax(gap_pred_y / tau, -1)
    softmax_gap_b = nn.functional.softmax(gap_batch_y / tau, -1)
    loss_gap = softmax_gap_p * \
        torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
    return loss_gap.mean()
    

class ConvTAU(pl.LightningModule):
    def __init__(self, params: ParamsConvTAU):
        super().__init__()        
        self.params = params
        
        self.mse = nn.MSELoss() # to focus on intra-frame-level differences
        self.kl_divergence = kullback_leibler_divergence # to focus on inter-frame-level differences

        self.encoder = Encoder(params)
        self.tau = TAU(params)
        self.decoder = Decoder(params)
        
    
    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        
        h, skip = self.encoder(x)
        BT, C_, H_, W_ = h.shape
        
        h = h.view(B, T*C_, H_, W_)
        tau_out = self.tau(h)
        tau_out = tau_out.view(B*T, C_, H_, W_)
        
        out = self.decoder(tau_out, skip)
        out = out.view(B, T, 1, H, W)
        
        return out


    def single_prediction(self, frames):
        frames = frames.unsqueeze(0).float()
        out = self(frames)
        return out.squeeze(0).squeeze(1).detach()

        
    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        out = self(x)

        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_mse_loss", mse_loss, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_epoch=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        out = self(x)

        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_mse_loss", mse_loss, on_epoch=True)
        self.log("validation_kl_loss", kl_loss, on_epoch=True)

        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float().unsqueeze(2)
        out = self(x)

        loss, mse_loss, kl_loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_mse_loss", mse_loss, on_epoch=True)
        self.log("test_kl_loss", kl_loss, on_epoch=True)

        return loss
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        return optimizer
    
    
    def loss(self, pred, y):
        mse_loss = self.mse(pred, y)
        kl_loss = self.params.kullback_leibler_divergence_weight * self.kl_divergence(pred, y)
        loss = mse_loss + kl_loss
        return (loss, mse_loss, kl_loss)
        
