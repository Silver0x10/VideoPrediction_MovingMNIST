

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
        self.training_epochs = 5
        
        self.learning_rate = 0.01

params_ConvTAU = {
    # Input parameters
    'frames_per_sample' : shared_params['MOVING_MNIST_INPUT_FRAMES'],
    'channels' : shared_params['CHANNELS'],
    'frame_height' : shared_params['HEIGHT'],
    'frame_width' : shared_params['WIDTH'],
    'in_shape' : (shared_params['MOVING_MNIST_INPUT_FRAMES'], shared_params['CHANNELS'], shared_params['HEIGHT'], shared_params['WIDTH']),

    # Model parameters
    'hid_S' : 32, #16, # hidden spatial representation size (encoder output, decoder input)
    'hid_T' : 128, #256, # hidden temporal representation size (internal to TAU module)
    'N_S' : 4, # number of spatial modules inside encoder and decoder
    'N_T' : 8, #4, # number of temporal modules inside the middle layer
    'mlp_ratio' : 8., # TODO understand what does it means exacly
    'drop' : 0.0, # TODO not used, remove
    'drop_path' : 0.0, # TODO not used, remove
    'spatio_kernel_enc' : 3, # kernel size of the encoder conv layers
    'spatio_kernel_dec' : 5, #3, # kernel size of the decoder conv layers
    # self.act_inplace = True
    
    # Training parameters
    'batch_size' : 16,
    'training_epochs' : 5,
    'learning_rate' : 0.01,
    'weight_decay' : 0.05,
    'kl_divergence_weight' : 0.5,
}
