import os
import torch
from torch import optim, nn, FloatTensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl


class SimpleLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss();
        self.relu = nn.ReLU()
        
        self.encoder = nn.Sequential(nn.Linear(4096, 2048), self.relu, nn.Linear(2048, 2048), self.relu)
        self.lstm = nn.LSTM(2048, 1024, bidirectional=True)     
        self.decoder = nn.Sequential(nn.Linear(2048, 2048), self.relu, nn.Linear(2048, 4096), self.relu, nn.Unflatten(1, (64,64)))

    def forward(self, x, h = None):
        if len(x.shape) < 3: # single frame prediction:
            frame = x.view(x.size(0)*x.size(1)).float()
            encoded_frame = self.encoder(frame).unsqueeze(0)
            lstm_out, h = self.lstm(encoded_frame, h)
            lstm_out = self.relu(lstm_out)
            out = self.decoder(lstm_out)
            return (out, h)
            
        # sequence prediction:
        x = x.unsqueeze(0)
        lstm_out = None
        h = None
        for frame_nr in range(x.shape[1]):
            frame = x[:, frame_nr, :, :].view(x.size(0), -1).float()
            encoded_frame = self.encoder(frame)
            lstm_out, h = self.lstm(encoded_frame, h)        
            lstm_out = self.relu(lstm_out)
        
        out = self.decoder(lstm_out).unsqueeze(1)
        
        for i in range(9):
            lstm_out, h = self.lstm(lstm_out, h)
            lstm_out = self.relu(lstm_out)
            out_i = self.decoder(lstm_out).unsqueeze(1)
            out = torch.cat((out, out_i), 1)

        return (out.squeeze(0), h)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        h = None
        lstm_out = None
        for frame_nr in range(x.shape[1]):
            frame = x[:, frame_nr, :, :].view(x.size(0), -1).float()
            encoded_frame = self.encoder(frame)
            lstm_out, h = self.lstm(encoded_frame, h)        
            lstm_out = self.relu(lstm_out)

        out = self.decoder(lstm_out).unsqueeze(1)
        
        for i in range(y.shape[1] - 1):
            lstm_out, h = self.lstm(lstm_out, h)
            lstm_out = self.relu(lstm_out)
            out_i = self.decoder(lstm_out).unsqueeze(1)
            out = torch.cat((out, out_i), 1)

        loss = self.loss(out, y)
        # self.log("mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        h = None
        lstm_out = None
        for frame_nr in range(x.shape[1]):
            frame = x[:, frame_nr, :, :].view(x.size(0), -1).float()
            encoded_frame = self.encoder(frame)
            lstm_out, h = self.lstm(encoded_frame, h)        
            lstm_out = self.relu(lstm_out)
        
        out = self.decoder(lstm_out).unsqueeze(1)
        
        for i in range(y.shape[1] - 1):
            lstm_out, h = self.lstm(lstm_out, h)
            lstm_out = self.relu(lstm_out)
            out_i = self.decoder(lstm_out).unsqueeze(1)
            out = torch.cat((out, out_i), 1)

        loss = self.loss(out, y)
        self.log("valid_loss", loss, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()

        h = None
        lstm_out = None
        for frame_nr in range(x.shape[1]):
            frame = x[:, frame_nr, :, :].view(x.size(0), -1).float()
            encoded_frame = self.encoder(frame)
            lstm_out, h = self.lstm(encoded_frame, h)        
            lstm_out = self.relu(lstm_out)
        
        out = self.decoder(lstm_out).unsqueeze(1)
        
        for i in range(y.shape[1] - 1):
            lstm_out, h = self.lstm(lstm_out, h)
            lstm_out = self.relu(lstm_out)
            out_i = self.decoder(lstm_out).unsqueeze(1)
            out = torch.cat((out, out_i), 1)

        loss = self.loss(out, y)
        self.log("test_loss", loss, on_epoch=True)
        return loss
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)