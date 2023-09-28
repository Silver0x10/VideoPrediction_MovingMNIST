import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

class OpticalFlowEstimator(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.estimator = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)

        for param in self.parameters(): param.requires_grad = False # For the moment, we don't want to train this model.

    def forward(self, prev_frame, curr_frame):
        prev_frame, curr_frame = self.preprocess(prev_frame, curr_frame)
        return self.estimator(prev_frame, curr_frame)
    
    def forward_sequence(self, frames):
        B, T, C, H, W = frames.shape
        
        curr_frames = frames.clone().view(B*T, C, H, W)
        curr_frames = torch.cat([curr_frames,curr_frames,curr_frames], 1).float()
        print(curr_frames.shape)
        
        print(frames.shape)
        prev_frames = torch.cat([torch.zeros((B, 1, C, H, W)), frames[:, 1:, :, :, :]], 1).view(B*T, C, H, W)
        prev_frames = torch.cat([prev_frames,prev_frames,prev_frames], 1).float()
        print(prev_frames.shape)
        
        return self.forward(prev_frames, curr_frames)

    @staticmethod
    def to_image(optical_flow):
        optical_flow = flow_to_image(optical_flow).squeeze(0)
        return F.to_pil_image(optical_flow.to("cpu"))
    
    def preprocess(self, prev_frame, curr_frame):
        prev_frame = F.resize(prev_frame, size=[720, 720], antialias=False)
        curr_frame = F.resize(curr_frame, size=[720, 720], antialias=False)
        return self.transforms(prev_frame, curr_frame)