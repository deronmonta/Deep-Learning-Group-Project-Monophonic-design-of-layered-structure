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

        #new_columns = ['R','T','A','d1','d2','d3','d4','d5','d6','d7','d8','Lambda']

        #self.dataframe = self.dataframe.reindex(columns=new_columns)

        print(self.dataframe)

        self.mode = mode
        
    def __getitem__(self,index):

        RT = self.dataframe.iloc[index,0:2].values 
        design_parameters = self.dataframe.iloc[index,2:]
        # normalized_design_parameters=(design_parameters-design_parameters.mean())/design_parameters.std()
        # normalized_design_parameters = normalized_design_parameters.values

        #Transform to pytorch tensor
        design_parameters = torch.tensor(design_parameters)
        RT = torch.tensor(RT)
        sample = {'RT':RT, 'design_parameters':design_parameters}

        if self.mode == 'gan':
            data = self.dataframe.iloc[index,:].values
            data = torch.tensor(data)
            sample = {'full_data':data}
            #print('sample from dataloader {}'.format(sample))
        return sample


    def __len__(self):
        return len(self.dataframe)
        
