# VideoPrediction_MovingMNIST

Prediction of future frames given past video frames.

<div align="center"> 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Silver0x10/VideoPrediction_MovingMNIST/blob/main/notebooks/VideoPredictionMovingMNIST.ipynb)

</div>

## Dataset 
[Moving MNIST](https://paperswithcode.com/dataset/moving-mnist). Introduced by Srivastava et al. in Unsupervised Learning of Video Representations using LSTMs. The Moving MNIST dataset contains 10,000 video sequences, each consisting of 20 frames. In each video sequence, two digits move independently around the frame, which has a spatial resolution of 64×64 pixels. The digits frequently intersect with each other and bounce off the edges of the frame. 

## Metrics 
Mean Squared Error.
Training sessions data available [here](https://wandb.ai/worst_dream_team/DeepLearning).

## SOTA
[Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning](https://paperswithcode.com/paper/temporal-attention-unit-towards-efficient)

<div align="center"> <img src="out/convTAU_comparison.png" width="70%"/> </div>

## Baseline Models

### SimpleLSTM

<div align="center"> <img src="out/simpeLSTM_comparison.png" width="70%"/> </div>

### Conv+LSTM

<div align="center"> <img src="out/convLSTM_comparison.png" width="70%"/> </div>
