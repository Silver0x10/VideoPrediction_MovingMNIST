
MOVING_MNIST_TOTAL_FRAMES = 20
CHANNELS = 1
HEIGHT = 64
WIDTH = 1

BATCH_SIZE = 32
MOVING_MNIST_INPUT_FRAMES = 10

class ParamsConvTAU():
    def __init__(self):
        self.learning_rate = 0.01
        self.weight_decay = 0.05
        self.batch_size = 5
        self.frames_per_sample = MOVING_MNIST_INPUT_FRAMES
        self.channels = 1
        
        self.kernel_size = 3
        
        self.dilation = 3
        self.fc_hidden_dim = 4
        
        self.kullback_leibler_divergence_weight = 0.5