import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
import pandas as pd
import numpy as np
import pickle

class Layer_Dataset(Dataset):
    '''
    Dimension is 3703197 by 12
    '''
    def __init__(self,data_dir,mode='regression'):
        print(data_dir)
        
        file = open(data_dir,'rb')
        self.dataframe  = pickle.load(file)
        self.dataframe = self.dataframe.dropna()
        print(self.dataframe)

        

        self.mode = mode
        
    def __getitem__(self,index):
        layer_thickness = self.dataframe.iloc[index,3:12].values # last 9 columns for layer thickness plus lambda
        RTA = self.dataframe.iloc[index,0:3].values
        
        #Transform to pytorch tensor
        layer_thickness_tensor = torch.tensor(layer_thickness)
        RTA = torch.tensor(RTA)
        sample = {'RTA':RTA, 'layer_thickness':layer_thickness_tensor}

        if self.mode == 'gan':
            data = self.dataframe.iloc[index,:].values
            data = torch.tensor(data)
            sample = {'full_data':data}
            #print('sample from dataloader {}'.format(sample))
        return sample


    def __len__(self):
        return len(self.dataframe)
        
