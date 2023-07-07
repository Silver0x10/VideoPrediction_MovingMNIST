# Per ora metto rtutti uguali ad 1 solo per poterli chiamare e scrivere il codice, poi vanno cambiati coerentemente con i valori del dataset

B = 1  # Batch size, quindi numero di video
T = 1  # n° di frames per video
C = 1  # n° di canali
H = 1  # Height
W = 1  # Width

mid_size = 1  # Dimension of smaller layers in Encoder/Decoder
out_size = 1  # Dimension of the output of the Encoder. It will be the input of TAU

MOVING_MNIST_TOTAL_FRAMES = 20
MOVING_MNIST_INPUT_FRAMES = 10

class ParamsConvTAU():
    def __init__(self):
        self.batch_size = 5
        self.frames_per_sample = MOVING_MNIST_INPUT_FRAMES
        self.channels = 1
        
        self.kernel_size = 3
        
        self.dilation = 3
        self.fc_hidden_dim = 4