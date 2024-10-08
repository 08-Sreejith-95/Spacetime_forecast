import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import datapreprocess
from datapreprocess import build_m4_sets_from_config, get_df_series_from_m4id, build_list_of_timesteps, create_timesteps_for_test
import math
import json
import matplotlib.pyplot as plt
import copy
from torch.utils.data import Dataset, DataLoader


data_configs = OmegaConf.load('ToyModel/Config/M4_config.yaml')






def build_data_frame_with_date(data_config,
                               Type = 'train'):
    data_dict = {}
    df, indices = build_m4_sets_from_config(data_config, 'train')
    series_nan = get_df_series_from_m4id(df, data_config).iloc[0].values[1:]
    series_cropped = [x for x in series_nan if not (isinstance(x, float) and math.isnan(x))]
    dates = build_list_of_timesteps(data_configs, series_cropped)
    #print(train_last_date)
    data_dict['data'] = series_cropped
    data_dict['time_steps'] = dates
    if Type == 'train':
        return pd.DataFrame(data_dict)
    elif Type == 'test':
        train_last_date = dates[-1]
        data_dict_test = {}
        test_df, indices = build_m4_sets_from_config(data_config, Type)
        test_series = get_df_series_from_m4id(test_df, data_config).iloc[0].values[1:]
        test_timesteps = create_timesteps_for_test(train_last_date, test_series, data_config)
        data_dict_test['data'] = test_series
        data_dict_test['time_steps'] = test_timesteps
         
        return pd.DataFrame(data_dict_test)
        

#print(build_data_frame_with_date(data_configs))

def build_data_samples(data_df, window_configs):
    data_set = {}
    lag = window_configs.lag 
    horizon = window_configs.horizon
    samples = [w.to_numpy() for w in data_df['data'].rolling(window = lag + horizon)][lag + horizon - 1:]#to get time series samples of length (l+h)(l to learn history and h to predict)
    time_step = [w for w in data_df['time_steps'].rolling(window = lag + horizon)][lag + horizon - 1:]
    data_set['data_samples'] = samples
    data_set['time_labels'] = time_step
    return data_set

def train_val_split(data_indices, val_ratio=0.1):#by default 90:10 train:val split
    train_ratio = 1 - val_ratio
    last_train_index = int(np.round(len(data_indices) * train_ratio))
    return data_indices[:last_train_index], data_indices[last_train_index:]
    
    
####  following statement enclosed inside the comment  ### are to be typed and executed in main. all of this are declared here to test the functions defined ####
window_config = OmegaConf.load('ToyModel/Config/window_configs.yaml')



    
    
    
#creating dataset x = [x1,x2......,x_l, 0, 0, 0,....0_l+h], y =[0, 0, 0,......,0_lag, h_l+1, h_l+2, ...., h_l+h] :-- a sample dim = (batch,l,d)(input dim for sequence data in pytorch).. for 1d d = 1 therefore it will be l,1

class M1MonthlyDataset(Dataset):
    def __init__(self, 
                 data:np.array,
                 lag:int,
                 horizon:int):
        super().__init__()
        self.data_x = torch.from_numpy(data).unsqueeze(-1).float()#the entire sequence:- a sample of x is not this. it will be of lag length l
        self.data_y = copy.deepcopy(self.data_x[:,-horizon:, :])#
        self.lag = lag
        self.horizon = horizon
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        x[-self.horizon:] = 0 #masking horizon terms to 0 so that input data x is of lag length
        return x, y ,(self.lag, self.horizon)
    
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x
    

  


       
def build_data_loader(data:np.array,
                      w_config,
                      shuffle = True,
                      **dataloader_kwargs: any):
    
   
    dataset = M1MonthlyDataset(data, w_config.lag, w_config.horizon)
    dataloader = DataLoader(dataset,
                            shuffle = shuffle,
                            **dataloader_kwargs)
    return dataloader
    
#####to main    

      