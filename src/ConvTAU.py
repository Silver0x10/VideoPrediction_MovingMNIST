
from typing import Any
import torch
from torch import optim, nn
import lightning.pytorch as pl

class TAU(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

    
    def forward(self, x):
        pass
    
        
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
        
        # TODO tau here
        
        self.decoder = Decoder(params)
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch['frames'].float(), batch['y'].float()
        B, T, H, W = x.shape
        x = x.view(B*T, 1, H, W)
        
        h, skip = self.encoder(x)
        
        out = self.decoder(h, skip)
        print(out.shape)
        out = out.view(B, T, 1, H, W)
        print(out.shape)

        loss = self.loss(out, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)
        
