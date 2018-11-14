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

    New design has 106 columns, the last three columns are RTA
    The input dimension is therefore 103
    '''
    def __init__(self,data_dir,mode='regression'):
        print(data_dir)
        
        file = open(data_dir,'rb')
        self.dataframe  = pickle.load(file)
        self.dataframe = self.dataframe.dropna()

        new_columns = ['R','T','A','d1','d2','d3','d4','d5','d6','d7','d8','Lambda']

        self.dataframe = self.dataframe.reindex(columns=new_columns)

        print(self.dataframe)

        self.mode = mode
        
    def __getitem__(self,index):
<<<<<<< HEAD
        design_parameters = self.dataframe.iloc[index,:-3].values # last 9 columns for layer thickness plus lambda
        RTA = self.dataframe.iloc[index,-4:-1].values
        
        #Transform to pytorch tensor
        design_parameters = torch.tensor(design_parameters)
        RTA = torch.tensor(RTA)
        sample = {'RTA':RTA, 'design_parameters':design_parameters}
=======
        layer_thickness = self.dataframe.iloc[index,3:].values # last 9 columns for layer thickness plus lambda
        RTA = self.dataframe.iloc[index,0:3].values
        
        #Transform to pytorch tensor
        layer_thickness_tensor = torch.tensor(layer_thickness)
        RTA = torch.tensor(RTA)
        sample = {'RTA':RTA, 'layer_thickness':layer_thickness_tensor}
>>>>>>> bf55591d0f2eb81b4507ae11d949aaecfd0b8bd5

        if self.mode == 'gan':
            data = self.dataframe.iloc[index,:].values
            data = torch.tensor(data)
            sample = {'full_data':data}
            #print('sample from dataloader {}'.format(sample))
        return sample


    def __len__(self):
        return len(self.dataframe)
        
