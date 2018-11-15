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


#Hyperparameters

parser = argparse.ArgumentParser(description='This script will train a network for the forward pass of the data')
parser.add_argument('--data_dir',default='./data/30sampled_designs.pkl',help='path to directory containing the layer designs and outputs')
parser.add_argument('--model_dir',default='./model',help='Directory to save and load the model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--model_name',default='./1024neuron.pkl')
parser.add_argument('--hidden_neurons',type=int,default=256)
parser.add_argument('--learning_rate',type=float,default=0.0001)
parser.add_argument('--epochs',type=int,default=100,help='Number of epochs to train')
parser.add_argument('--in_dim',type=int,default=13,help='Input dimension')
parser.add_argument('--out_dim',type=int,default=2,help='Output dimension')

options = parser.parse_args()
print(options)



layer_dataset = Layer_Dataset(options.data_dir) 
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,num_workers=2)

dense_net = (Dense_Net(in_dim=options.in_dim,out_dim=options.out_dim,num_units=options.hidden_neurons)).cuda()
net_optimizer = optim.Adam(dense_net.parameters(),lr=options.learning_rate)

loss_func = nn.MSELoss()

if not os.path.exists(options.model_dir):
    os.mkdir(options.model_dir)

try:
    dense_net = torch.load(os.path.join(options.model_dir, options.model_name))
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")

for epoch in tqdm(range(options.epochs)):
    for i,sample in tqdm(enumerate(layer_dataset)):
        
        
        design_parameters = sample['design_parameters'].float().cuda()
        ground_truth = sample['RT'].float().cuda()
        predictions = dense_net(design_parameters)
        loss = loss_func(predictions,ground_truth)
        loss.backward() 
        net_optimizer.step()

        
        if i % 10000 == 0:
            print('\n')
            print('Epoch: {}'.format(epoch))
            print('Ground Truth: {}'.format(ground_truth.data.cpu()))
            print('Predictions {}'.format(predictions.data.cpu()))
            print('Loss: {}'.format(loss))
            torch.save(dense_net,os.path.join('./model',options.model_name))
        





    
