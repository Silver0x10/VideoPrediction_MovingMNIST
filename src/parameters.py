

shared_params = {
    'MOVING_MNIST_TOTAL_FRAMES': 20,
    'MOVING_MNIST_INPUT_FRAMES': 10,
    'CHANNELS': 1,
    'HEIGHT': 64,
    'WIDTH': 64
}

class ParamsSimpleLSTM():
    def __init__(self):
        self.frames_per_sample = shared_params['MOVING_MNIST_INPUT_FRAMES']
        self.frames_to_predict = 10

        self.batch_size = 16
        self.training_epochs = 5
        
        self.learning_rate = 0.01
        
class ParamsConvLSTM():
    def __init__(self):
        self.frames_per_sample = shared_params['MOVING_MNIST_INPUT_FRAMES']

        self.batch_size = 16
        self.training_epochs = 50
        
        self.learning_rate = 0.01

class ParamsConvTAU():
    def __init__(self):
        self.channels = shared_params['CHANNELS']
        self.frames_per_sample = shared_params['MOVING_MNIST_INPUT_FRAMES']

        self.batch_size = 16
        self.training_epochs = 5
        
        self.learning_rate = 0.01
        self.weight_decay = 0.05
        
        self.kernel_size = 3
        
        self.dilation = 3
        self.fc_hidden_dim = 4
        
        self.kullback_leibler_divergence_weight = 0.5