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
<<<<<<< HEAD
parser.add_argument('--model_name',default='./forward_model')
parser.add_argument('--hidden_neurons',type=int)

options = parser.parse_args()
print(options)


#Hyperparameters
NUM_NEURONS = 128
LEARNING_RATE = 0.0001
MODEL_NAME = 'forward_model.pkl'
EPOCHS = 20
=======
parser.add_argument('--model_name',default='./forward_model.pkl')
parser.add_argument('--hidden_neurons',type=int,default=256)
parser.add_argument('--learning_rate',type=float,default=0.0001)
parser.add_argument('--epochs',type=int,default=100,help='Number of epochs to train')

options = parser.parse_args()
print(options)
>>>>>>> 97bbe8628d274b09adc8a6224cde090264140b8c


#Hyperparameters

<<<<<<< HEAD
layer_dataset = Layer_Dataset(options.data_dir)
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,num_workers=2)
=======



layer_dataset = Layer_Dataset(options.data_dir) 
data_loader = DataLoader(layer_dataset, batch_size=options.data_dir,shuffle=True,num_workers=2)
>>>>>>> 97bbe8628d274b09adc8a6224cde090264140b8c

dense_net = (Dense_Net(in_dim=8,out_dim=4,num_neurons=options.hidden_neurons)).cuda()

net_optimizer = optim.Adam(dense_net.parameters(),lr=options.learning_rate)

loss_func = nn.MSELoss()

if not os.path.exists('./model'):
    os.mkdir('./model')

try:
    dense_net = torch.load(os.path.join('./model',options.model_name))
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")



for epoch in range(options.e):
    for i,sample in tqdm(enumerate(layer_dataset)):
        
        
        layer_thickness = sample['layer_thickness'].float().cuda()
        ground_truth = sample['Lambda_RTA'].float().cuda()
        
        dense_net.zero_grad()

        predictions = dense_net(layer_thickness)
        loss = loss_func(predictions,ground_truth)
        loss.backward()
        net_optimizer.step()

        if i % 1000 == 0:
            print('\n')
            print('Epoch: {}'.format(epoch))
            print('Ground Truth: {}'.format(ground_truth.data.cpu()))
            print('Predictions {}'.format(predictions.data.cpu()))
            print('Loss: {}'.format(loss))
            torch.save(dense_net,os.path.join('./model',options.model_name))
        





    
