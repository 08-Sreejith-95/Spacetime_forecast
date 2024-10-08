#code for seting up the time series data from m4 dataset. the dataset contains around 40000 time series for hourly, daily,mothly and yearly time steps
#the series are classified by their m4ids. and are rows in the pd dataframe
#currently only monthly data is downloaded and added to the data directory. dwnld and add other data for experimenting




import torch
import pandas as pd  
import torch.nn as nn
import os
import sys
from argparser import init_args
from datetime import datetime
from dateutil.relativedelta import relativedelta



#setup the
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #This is added since the python interpreter cant find the utils module from the root dir. So root dir is added in this script
from omegaconf import OmegaConf
from utils.logging import print_config


def build_m4_sets_from_config(config,
                    Type = 'train',
                    ):  
    slice_idxs = get_series_indices(config)
    #print("Available series ids:", slice_idxs)  
    if Type == 'train':
        df_train = pd.read_csv(config.train_path)
        new_df_set = get_df_from_idxs(df_train, slice_idxs)
    elif Type == 'test':
        df_test = pd.read_csv(config.test_path)
        new_df_set = get_df_from_idxs(df_test, slice_idxs)
    
    return new_df_set, slice_idxs

    


def get_df_from_idxs(df, idx_list):
     mask = df['V1'].isin(idx_list)
     new_df = df[mask]
     return new_df
 

def get_df_series_from_m4id(df, config):
    return df.loc[df['V1'] == config.M4id]
    
     
     
def get_series_indices(config):
     m4_info = pd.read_csv(config.label_path)
     m4_idxs = []
     for cat, ts, idx in  zip(m4_info['category'], m4_info['SP'], m4_info['M4id']) :
        if cat == config.category and ts == config.timestep:
            m4_idxs.append(idx)
     return m4_idxs
        
        
#to extract start time from the m4info for a time series: to do:- write code for daily, yearly and hourly data        
def get_start_time_from_df_train(config):
    df_labels = pd.read_csv(config.label_path)
    time_step = config.timestep
    df_series = df_labels.loc[df_labels['M4id'] == config.M4id]
    if time_step == 'Monthly':
        start_time = df_series['StartingDate']
    return start_time

#to get the list of timesteps for the series: todo:- write statements for daily, hourly and yearly data
def build_list_of_timesteps(config, np_array:list):
    dates_list = []
    n = len(np_array)
    if config.timestep == 'Monthly':
        start = get_start_time_from_df_train(config).values[0]
        start_time = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        for i in range(n):
            nxt_step = start_time + relativedelta(months = i)
            nxt_step = nxt_step.strftime('%Y-%m-%d %H:%M:%S')
            dates_list.append(nxt_step)
        
    return dates_list
        

def create_timesteps_for_test(train_last_ts, test_set, config):
    length_test = len(test_set)
    start_test = str(train_last_ts)
    print(train_last_ts)
    print('start_test ', start_test)
    time_list = []
    if config.timestep == 'Monthly':
        start_t = datetime.strptime(start_test, '%Y-%m-%d %H:%M:%S')
        print('train_end_time', start_t)
        for i in range(length_test):
            nxt_step = start_t + relativedelta(months = i)
            nxt_step = nxt_step.strftime('%Y-%m-%d %H:%M:%S')
            time_list.append(nxt_step)
        
 
    return time_list
    
    
    

    
    
    
           



        
        
        




    

