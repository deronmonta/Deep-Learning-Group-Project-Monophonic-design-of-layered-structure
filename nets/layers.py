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

def dense(input_dim, output_dim, act_fn=nn.Sigmoid()):
    '''
    Fully connected layer with no activation
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