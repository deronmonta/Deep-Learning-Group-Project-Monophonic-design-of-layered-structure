import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from dataset.layer_dataset import *
from nets.nets import *
from tqdm import tqdm
import argparse
#This trains the model with the forward pass

parser = argparse.ArgumentParser(description='This script will train a network for the forward pass of the data')
parser.add_argument('--data_dir',default='./data/all_design.csv',help='path to directory containing the layer designs and outputs')
parser.add_argument('--model_dir',default='./model',help='Directory to save and load the model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--model_name',default='./forward_model.pkl')
parser.add_argument('--hidden_neurons',type=int,default=256)
parser.add_argument('--learning_rate',type=float,default=0.0001)
parser.add_argument('--epochs',type=int,default=100,help='Number of epochs to train')

options = parser.parse_args()
print(options)

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,4,8,9"
    print('Manual GPU loading successful')  
    print('__Number CUDA Devices:', torch.cuda.device_count())
except:
    print('Manual GPU loading failed')
    pass

layer_dataset = Layer_Dataset(options.data_dir,mode='gan') 
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,num_workers=2)

dense_net = Dense_Net(in_dim=8,out_dim=4,num_neurons=options.hidden_neurons).cuda()
discriminator = 



net_optimizer = optim.Adam(dense_net.parameters(),lr=options.learning_rate)

loss_func = nn.MSELoss()

