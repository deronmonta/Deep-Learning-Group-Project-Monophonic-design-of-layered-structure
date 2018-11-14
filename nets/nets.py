import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from nets.layers import *

class Dense_Net(nn.Module):
    def __init__(self, in_dim, out_dim, num_units):
        """Fully connected network for forward regression
        
        Arguments:
            in_dim {int} -- [Number of input dimension]
            out_dim {int} -- [description]
            num_units {int} -- [description]
        """
        print('\n---------------------Dense Net Starting ---------------------')
        super(Dense_Net,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_units = num_units

        self.dense1 = dense(self.in_dim,self.num_units*1)
        self.dense2 = dense(self.num_units*1,self.num_units*2)
        self.dense3 = dense(self.num_units*2,self.num_units*4)
        self.dense4 = dense(self.num_units*4,self.num_units*8)
        self.dense5 = dense(self.num_units*8,self.out_dim,act_fn=None)#With no activation function

    
    def forward(self,input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x

class Generator(nn.Module):
    def __init__(self,in_dim,out_dim,num_units):

        super(Generator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_units = num_units

        self.dense1 = dense(self.in_dim,self.num_units*8)
        self.dense2 = dense(self.num_units*8,self.num_units*4)
        self.dense3 = dense(self.num_units*4,self.num_units*2)
        self.dense4 = dense(self.num_units*2,self.num_units*1)
        self.dense5 = dense(self.num_units*1,self.out_dim,act_fn=None)# Last layer with no activation function so that we get values above 1 
    
    def forward(self,input):

        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x 


class Discriminator(nn.Module):
    def __init__(self,in_dim,out_dim,num_units):
        
        super(Discriminator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_units = num_units

        self.dense1 = dense(self.in_dim,self.num_units*1)
        self.dense2 = dense(self.num_units*1,self.num_units*2)
        self.dense3 = dense(self.num_units*2,self.num_units*4)
        self.dense4 = dense(self.num_units*4,self.num_units*8)
        self.dense5 = dense(self.num_units*8,self.out_dim)
    
    def forward(self,input):

        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return x 



    
    