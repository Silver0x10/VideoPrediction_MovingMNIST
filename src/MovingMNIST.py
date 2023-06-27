import pytorch_lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
import torch
import io
import imageio
from ipywidgets import widgets, HBox
from IPython.display import display
import matplotlib as plt


class MovingMNIST(Dataset):
    def __init__(self, data_path = 'data/mnist_test_seq.npy'):
        super().__init__()
        self.data_path = data_path
        self.data = torch.from_numpy(np.load(self.data_path).transpose(1, 0, 2, 3))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if index >= len(self): return None
        video = self.data[index]
        return {'frames' : video[:-1], 'y' : video[-1]}
    
    def visualize(self, index):
        video = self[index]['frames']
        y = self[index]['y']
        with io.BytesIO() as gif:
            imageio.mimsave(gif,video.numpy().astype(np.uint8),"GIF",fps=5)
            display(HBox([widgets.Image(value=gif.getvalue())]))


# class MovingMNIST(pl.LightningDataModule):
#     def __init__(self, data_path = 'data/mnist_test_seq.npy', batch_size = 8):
#         super().__init__()
        
#     def setup(self):