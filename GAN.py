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
parser.add_argument('--data_dir',default='./data/all_design.pkl',help='path to directory containing the layer designs and outputs')
parser.add_argument('--model_dir',default='./model',help='Directory to save and load the model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--G_model_name',default='Generator_model.pkl')
parser.add_argument('--D_model_name',default='Discriminator_model.pkl')
parser.add_argument('--hidden_neurons',type=int)

options = parser.parse_args()
print(options)

layer_dataset = Layer_Dataset(options.data_dir)
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,num_workers=2)

G = (Generator(in_dim=8,out_dim=4,num_neurons=NUM_NEURONS)).cuda()
D = (Discriminator(in_dim=8,out_dim=4,num_neurons=NUM_NEURONS)).cuda()


net_optimizer = optim.Adam(dense_net.parameters(),lr=LEARNING_RATE)
loss_func = nn.MSELoss()
