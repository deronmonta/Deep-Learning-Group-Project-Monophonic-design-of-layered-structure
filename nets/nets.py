import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *

class Dense_Net(nn.Module):
    def __init__(self, in_dim, out_dim, num_neurons):
        '''
        '''
        print('\n---------------------Dense Net---------------------')
        super(Dense_Net,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neurons = num_neurons

        self.dense1 = dense_activation(in_dim,self.num_neurons*1)
        self.dense2 = dense_activation(self.num_neurons*1,self.num_neurons*2)
        self.dense3 = dense_activation(self.num_neurons*2,self.num_neurons*4)
        self.dense4 = dense_activation(self.num_neurons*4,self.num_neurons*8)
        self.dense5 = dense(self.num_neurons*8,self.output_dim)

    
    def forward(self,input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x

        


