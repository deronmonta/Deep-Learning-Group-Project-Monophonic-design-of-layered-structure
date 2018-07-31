import torch
from torch.utils.data import Dataset,DataLoader
#import torchbiomed.utils as utils
from glob import glob
import os
import pandas as pd
import numpy as np


class Layer_Dataset(Dataset):
    '''
    Dimension is 3703197 by 12
    '''
    def __init__(self,data_dir):
        print(data_dir)
        
        self.dataframe = pd.read_pickle(data_dir)
        print(self.dataframe)


    def __getitem__(self,index):
        layer_thickness = self.dataframe.iloc[index,4:12] # last 8 columns for layer thickness
        Lambda_RTA = self.dataframe.iloc[index,0:4].values
        

        print(layer_thickness)
        #Transform to numpy array
        layer_thickness_np = layer_thickness.values

        sample = {'Lambda_RTA':Lambda_RTA 'layer_thickness':layer_thickness_np}
        return sample
    def __len__(self):
        return len(self.dataframe)
        
