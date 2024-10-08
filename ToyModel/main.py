import sys
import os
from os.path import join
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(parent_dir)
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))

import torch
import numpy as np
from datapreprocess import build_m4_sets_from_config, get_df_series_from_m4id, build_list_of_timesteps, create_timesteps_for_test
from dataloader import build_data_frame_with_date, build_data_samples, train_val_split, build_data_loader, M1MonthlyDataset
from toyconfigs import init_encoder_decoder_config, TrainConfigs
from model.network import SpaceTime
from setup import seed_everything
from omegaconf import OmegaConf
from utils.config import print_config
import pandas as pd

#import training modules
from loss import get_loss
from data_transforms import get_data_transforms
from optimizer import get_optimizer, get_scheduler
from setup.configs.optimizer import get_optimizer_config, get_scheduler_config
from train import train_model, evaluate_model, plot_forecasts

#train loop
from setup import initialize_experiment




main_config_dir = 'configs'
##initializing all the configurations
#----data configs-----#
data_configs = OmegaConf.load('ToyModel/Config/M4_config.yaml')
window_config = OmegaConf.load('ToyModel/Config/window_configs.yaml')
data_loader_configs = OmegaConf.load('ToyModel/Config/data_loader_configs.yaml')



#----model configs------#
embedd_config_toy = OmegaConf.load('ToyModel/Config/model_configs/embedding_config_toy.yaml')
encoder_config_toy = OmegaConf.load('ToyModel/Config/model_configs/encoder_config_toy.yaml')
decoder_config_toy = OmegaConf.load('ToyModel/Config/model_configs/decoder_config_toy.yaml')
output_config_toy = OmegaConf.load('ToyModel/Config/model_configs/output_config_toy.yaml')

#creating the network from configs
encoder_config = init_encoder_decoder_config(encoder_config_toy, join(main_config_dir, 'model'))
decoder_config = init_encoder_decoder_config(decoder_config_toy, join(main_config_dir, 'model'))

#Training hyperparameter configuration
train_configs = OmegaConf.load('ToyModel/Config/train_configs.yaml')

#initialize our SpaceTime model
model_configs = {
    'embedding_config': embedd_config_toy,
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'output_config': output_config_toy,
    'lag': window_config.lag,
    'horizon': window_config.horizon
}
seed_everything(data_configs.seed)

model = SpaceTime(**model_configs)
print('|---------------- Model Architecture -----------------|')
print_config(model_configs)


#Spacetime network training
train_args = TrainConfigs(train_configs)
#extra configurations
train_args.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
train_args.dataset_type = 'informer'
train_args.dataset = 'M14'

train_args.checkpoint_dir = './checkpoints'
train_args.log_dir = './log_dir'
train_args.variant = None
train_args.no_wandb = True
train_args.dataset_type = 'informer'  # for standard forecasting
train_args.log_epoch = 1000
train_args.features = 'S'
train_args.horizon = window_config.horizon
train_args.lag = window_config.lag


#----initializing training loop components

#general setup
seed_everything(train_configs.seed)
model = SpaceTime(**model_configs)

#optimizer and scheduler
optimizer = get_optimizer(model, get_optimizer_config(train_configs, main_config_dir))
scheduler = get_scheduler(model, optimizer, get_scheduler_config(train_configs, main_config_dir))

#setting up the loss function
criterions = {name:get_loss(name) for name in ['rmse', 'mse', 'mae']}
eval_criterions = criterions
for name in ['rmse', 'mse', 'mae']:
    eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')
    
#basic data transforms- Normalization
input_transform, output_transform = get_data_transforms(train_configs.data_transform, window_config.lag)


#setting up the dataloaders
data_df_train = build_data_frame_with_date(data_configs, Type = 'train')
data_df_test = build_data_frame_with_date(data_configs, Type = 'test')
data_train= build_data_samples(data_df_train, window_config)
data_samples_train = data_train['data_samples']
time_labels_train = data_train['time_labels']


#spliting to train and val
indices = np.arange(len(data_samples_train))
train_indices, val_indices = train_val_split(indices)
train_samples = np.array(data_samples_train[:val_indices[0]])
train_time_labels = np.array(time_labels_train[:val_indices[0]])
val_samples = np.array(data_samples_train[val_indices[0]:])
val_time_labels = np.array(time_labels_train[val_indices[0]:])

#setting up test set
last_samples = val_samples[-1][-window_config.lag:]
test_series_lag_time = val_time_labels[-1][-window_config.lag:]
new_lag_df =pd.DataFrame({'data':last_samples, 'time_steps':test_series_lag_time})

new_df_test = pd.concat([new_lag_df, data_df_test],axis=0, ignore_index=False)
data_test = build_data_samples(new_df_test, window_config)
test_samples = np.array(data_test['data_samples'], dtype=np.float32)
time_labels_test = np.array(data_test['time_labels'])


#print('train_samples', train_samples[-1])
#print('test_samples', data_test)

train_loader = build_data_loader(train_samples, window_config)
val_loader = build_data_loader(val_samples, window_config, shuffle = False)
test_loader = build_data_loader(test_samples, window_config, shuffle=False)





#--------------Training the model----------------#
initialize_experiment(train_args,
                      experiment_name_id = 'Toymodel_with m14',
                      best_train_metric = 1e10,
                      best_val_metric = 1e10
                     )
#building dataloaders
dataloaders_by_split ={'train': train_loader,
              'val': val_loader,
              'test': test_loader}
#training loop
model = train_model(model, optimizer, scheduler, dataloaders_by_split, criterions, max_epochs = train_args.max_epochs, config=train_args,
                    input_transform = input_transform,
                    output_transform = output_transform,
                    val_metric = train_args.val_metric,
                    wandb = None,
                    return_best = True,
                    early_stopping_epochs = train_args.early_stopping_epochs)




