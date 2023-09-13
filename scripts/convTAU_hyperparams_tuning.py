
import sys, os
# file_dir = os.path.dirname()
# sys.path.append('../VideoPrediction_MovingMNIST/')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
# from lightning.pytorch.loggers import WandbLogger
# import wandb

from src.MovingMNIST import MovingMNIST
from src.ConvTAU import ConvTAU
from src.parameters import params_ConvTAU, shared_params

tuned_values = {
    'hid_S' : [16, 32 , 64],
    'hid_T' : [128, 256, 512],
    'N_S' : [2, 4, 8],
    'N_T' : [2, 4, 8],
    'spatio_kernel_enc' : [3, 5, 7], 
    'spatio_kernel_dec' : [3, 5, 7], 
}

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MovingMNIST(data_path="data/mnist_test_seq.npy")
train_set, validation_set, test_set = random_split(dataset,[8000, 1000, 1000],
                                                   generator=torch.Generator().manual_seed(42))
training_dataloader = DataLoader(train_set, batch_size = params_ConvTAU['batch_size'])
validation_dataloader = DataLoader(validation_set, batch_size = params_ConvTAU['batch_size'])
test_dataloader = DataLoader(test_set, batch_size = params_ConvTAU['batch_size'])

results = []

for key in tuned_values.keys():
    for val in tuned_values[key]:
        print("##########", key, val, "##########")
        tuned_params = params_ConvTAU.copy()
        tuned_params[key] = val

        model_convTAU = ConvTAU(tuned_params)
        
        test_name = "ConvTAU_" + key + str(val)
        # wandb_logger = WandbLogger(project='DeepLearning', name=test_name)
        
        trainer = pl.Trainer(max_epochs=3, accelerator=device.type) # logger=wandb_logger
        trainer.fit(model=model_convTAU, train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)
        
        loss = trainer.test(model_convTAU, dataloaders=test_dataloader)
        
        res = {}
        res['loss'] = loss
        res['param'] = key
        res['value'] = val
        results.append(res)
        
for test in results:
    out = str(test['param']) + "\t" + str(test['value']) + "\t" + str(test['loss']) + "\n"
    print(out)
    
        