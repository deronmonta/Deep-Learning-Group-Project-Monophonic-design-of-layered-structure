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
        self.dense3 = dense(self.num_units*2,self.out_dim)
        # self.dense4 = dense(self.num_units*4,self.num_units*8)
        # self.dense5 = dense(self.num_units*8,self.num_units*16)
        # self.dense6 = dense(self.num_units*16,self.out_dim)#With no activation function

    
    def forward(self,input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        # x = self.dense4(x)
        # x = self.dense5(x)
        # x = self.dense6(x)

        return x


class Dense_Net_Residual(nn.Module):
    def __init__(self,in_dim, out_dim, num_units):
        """[summary]
        
        Arguments:
            nn {[type]} -- [description]
            in_dim {[type]} -- [input dimension]
            out_dim {[type]} -- [output dimension]
            units {[type]} -- [number of first layer neurons, each layer doubles the number of neurons]
        """
        super(Dense_Net_Residual,self).__init__()
        print('\n---------------------Dense Net with Residual Starting ---------------------')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.units = num_units
        
        self.fc1 = Dense_Block(in_dim, self.units)
        self.fc2 = Dense_Block(self.units,self.units*2)
        self.fc3 = Dense_Block(self.units*2,self.units*4)
        self.fc4 = Dense_Block(self.units*4,out_dim,act_fn=None)

    def forward(self,input):
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)

        return out

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



    
    