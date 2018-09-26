import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from nets.layers import *

class Dense_Net(nn.Module):
    def __init__(self, in_dim, out_dim, num_neurons):
        '''
        Fully connected network for forward regression
        '''
        print('\n---------------------Dense Net Starting ---------------------')
        super(Dense_Net,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neurons = num_neurons

        self.dense1 = dense_activation(self.in_dim,self.num_neurons*1)
        self.dense2 = dense_activation(self.num_neurons*1,self.num_neurons*2)
        self.dense3 = dense_activation(self.num_neurons*2,self.num_neurons*4)
        self.dense4 = dense_activation(self.num_neurons*4,self.num_neurons*8)
        self.dense5 = dense(self.num_neurons*8,self.out_dim)

    
    def forward(self,input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x

        
class Generator(nn.Module):
    """[summary]
    
    Args:
        nn ([type]): [description]
    """
    def __init__(self,in_dim,out_dim,nf):
        super(Generator,self).__init__()
        






class Discriminator(nn.Module):
    