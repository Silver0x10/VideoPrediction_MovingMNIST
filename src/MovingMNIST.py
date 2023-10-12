import lightning as pl
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
import torch
import io
import imageio
from ipywidgets import widgets, HBox
from IPython.display import display
import matplotlib as plt

from src.parameters import shared_params


class MovingMNIST(Dataset):
    def __init__(self, data_path='data/mnist_test_seq.npy'):
        super().__init__()
        self.data_path = data_path
        self.data = torch.from_numpy(
            np.load(self.data_path).transpose(1, 0, 2, 3))
        self.input_frames = shared_params['MOVING_MNIST_INPUT_FRAMES']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            return None
        video = self.data[index]
        return {'index': index, 'frames': video[:self.input_frames], 'y': video[self.input_frames:]}

    def visualize_as_gif(self, index):
        video = torch.concat((self[index]['frames'], self[index]['y']))
        with io.BytesIO() as gif:
            imageio.mimsave(gif, video.numpy().astype(np.uint8), "GIF", fps=5)
            display(HBox([widgets.Image(value=gif.getvalue())]))

    def visualize_given_frames_as_gif(self, frames):
        frames = frames.float()
        with io.BytesIO() as gif:
            imageio.mimsave(gif, frames.numpy().astype(np.uint8), "GIF", fps=5)
            display(HBox([widgets.Image(value=gif.getvalue())]))
