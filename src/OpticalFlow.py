import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["savefig.bbox"] = "tight"

# def to_image(optical_flow):
#     optical_flow = flow_to_image(optical_flow).squeeze(0)
#     return F.to_pil_image(optical_flow.to("cpu"))

# def plot(imgs, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             img = F.to_pil_image(img.to("cpu"))
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     plt.tight_layout()

class OpticalFlowEstimator(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.estimator = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        
        self.resolution = 128

        for param in self.parameters(): param.requires_grad = False

    def forward(self, prev_frame, curr_frame):
        prev_frame, curr_frame = self.preprocess(prev_frame, curr_frame)
        return self.estimator(prev_frame, curr_frame)
    
    def forward_sequence(self, frames):
        B, T, C, H, W = frames.shape
        flows = [torch.zeros((B, 2, self.resolution, self.resolution), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]
        prev_frames = torch.cat([frames[:, 0, :], frames[:, 0, :], frames[:, 0, :]], 1).float()
        for i in range(1, T):
            curr_frames = torch.cat([frames[:, 1, :], frames[:, 1, :], frames[:, i, :]], 1).float()
            flow = self.forward(prev_frames, curr_frames)[-1]
            flows.append(flow)
            prev_frames = curr_frames
        
        return torch.cat(flows)
        

    def preprocess(self, prev_frame, curr_frame):
        prev_frame = F.resize(prev_frame, size=[self.resolution, self.resolution], antialias=False)
        curr_frame = F.resize(curr_frame, size=[self.resolution, self.resolution], antialias=False)
        return self.transforms(prev_frame, curr_frame)
    
    
class OpticalLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.optical_flow_estimator = OpticalFlowEstimator()
        self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU()


    def forward(self, x):
        flows = self.optical_flow_estimator.forward_sequence(x)
        flows = self.activation(self.conv(flows)) # downsample
        return flows
    
    