import os
import torch
from torch import optim, nn, FloatTensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# movingMNIST = MNIST('../data', train=True, transform=ToTensor, download=True)
# train_loader = utils.data.DataLoader(movingMNIST, batch_size=8, shuffle=True)

class simpleLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss();
        self.relu = nn.ReLU()
        
        self.encoder = nn.Sequential(nn.Linear(4096, 1024), self.relu)
        self.lstm = nn.LSTM(1024, 512)     
        self.decoder = nn.Sequential(nn.Linear(512, 1024), self.relu, nn.Linear(1024, 4096), self.relu, nn.Unflatten(1, (64,64)))

    def forward(self, x):
        # TODO something like the training step
        return torch.zeros((1, 64,64))
    
    def training_step(self, batch, batch_idx):
        x, y = batch['frames'], batch['y'].float()
        
        h = None
        lstm_out = None
        for frame_nr in range(x.shape[1]):
            frame = x[:, frame_nr, :, :].view(x.size(0), -1).float()
            encoded_frame = self.encoder(frame)
            lstm_out, h = self.lstm(encoded_frame, h)
        lstm_out = self.relu(lstm_out)
        out = self.decoder(lstm_out)

        loss = self.loss(out, y)
        self.log("mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    
    # def test_step(self, batch, batch_idx):
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)