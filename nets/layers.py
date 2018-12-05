import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

def dense_activation(input_dim,output_dim):
    '''
    Fully connected layer with activation
    Args:
        input_dim:
        output_dim:
    '''
    layer = nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.Sigmoid()
    )

    return layer

def dense(input_dim, output_dim, act_fn=nn.ReLU()):
    '''
    Fully connected layer w/o activation
    Args:
        input_dim
        output_dim
    '''
    if act_fn is not None:

        layer = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            act_fn
        )

    else:
        layer = nn.Sequential(
            nn.Linear(input_dim,output_dim)
        )


    return layer


class Dense_Block(nn.Module):

    def __init__(self,indim, outdim,act_fn=nn.Sigmoid()):
        super(Dense_Block, self).__init__()
        self.act_fn = act_fn
        self.indim = indim
        self.outdim = outdim
        self.fc = nn.Linear(self.indim,self.outdim)
        self.act_fn = act_fn


    
    def forward(self,input_):

        if self.act_fn is not None:

            residual = input_
            out = self.fc(input_)
            out = self.act_fn(out)
            out += residual # Residual connection here
            out = self.act_fn(out)

            return out
        
        else:
            
            out = self.fc(input_)

            return out