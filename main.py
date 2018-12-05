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
from logger import Logger
import numpy as np 
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)



#This trains the model with the forward pass


#Hyperparameters

parser = argparse.ArgumentParser(description='This script will train a network for the forward pass of the data')
parser.add_argument('--data_dir',default='./data/30sampled_designs.pkl',help='path to directory containing the layer designs and outputs')
parser.add_argument('--model_dir',default='./model',help='Directory to save and load the model')
parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('--model_name',default='./model_3_layer.pkl')
parser.add_argument('--hidden_neurons',type=int,default=1024)
parser.add_argument('--learning_rate',type=float,default=0.0001)
parser.add_argument('--epochs',type=int,default=100,help='Number of epochs to train')
parser.add_argument('--in_dim',type=int,default=13,help='Input dimension')
parser.add_argument('--out_dim',type=int,default=2,help='Output dimension')
parser.add_argument('--log_dir',default='./logs',help='directory to save the logs')

options = parser.parse_args()
print(options)

if not os.path.exists(options.model_dir):
    os.mkdir(options.model_dir)
if not os.path.exists(options.log_dir):
    os.mkdir(options.log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = Logger(options.log_dir)

layer_dataset = Layer_Dataset(options.data_dir) 
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,num_workers=8)


dense_net = nn.DataParallel(Dense_Net(in_dim=options.in_dim,out_dim=options.out_dim,num_units=options.hidden_neurons)).cuda()

try:
    dense_net = torch.load(os.path.join(options.model_dir, options.model_name))
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")


net_optimizer = optim.Adam(dense_net.parameters(),lr=options.learning_rate,amsgrad=True)
#net_optimizer = optim.SparseAdam(dense_net.parameters(),lr=options.learning_rate)
loss_func = nn.MSELoss()

for epoch in tqdm(range(options.epochs)):
    for i,sample in tqdm(enumerate(data_loader)):
        
        
        design_parameters = sample['design_parameters'].float().cuda()
        
        ground_truth = sample['RT'].float().cuda()
        predictions = dense_net(design_parameters)
        loss = loss_func(predictions,ground_truth)
        loss.backward() 
        net_optimizer.step()
        net_optimizer.zero_grad()
        
        if i  % 1000 == 0:
            print('\n')
            print('Epoch: {}'.format(epoch))
            print('Ground Truth: {}'.format(ground_truth.data.cpu().numpy()))
            print('Predictions: {}'.format(predictions.data.cpu().numpy()))
            print('Loss: {}'.format(loss))
            torch.save(dense_net,os.path.join('./model',options.model_name))


            #######################################################################################################################
            # Tensorboard logger
            #######################################################################################################################
            
            #info = {'loss': loss.item()}


            # for tag,value in info.items():
            #     logger.scalar_summary(tag,value,i+1)


            # for tag, value in dense_net.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
            #     logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)









    
