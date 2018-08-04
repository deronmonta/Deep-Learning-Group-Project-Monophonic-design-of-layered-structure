import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from dataset.layer_dataset import *
from nets.nets import *
from tqdm import tqdm

#This trains the model with the forward pass


#Hyperparameters
DATA_DIR = './data/all_design.csv'
BATCH_SIZE = 64
NUM_NEURONS = 128
LEARNING_RATE = 0.0001
MODEL_NAME = 'forward_model.pkl'
EPOCHS = 20






layer_dataset = Layer_Dataset(DATA_DIR) 
data_loader = DataLoader(layer_dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

dense_net = (Dense_Net(in_dim=8,out_dim=4,num_neurons=NUM_NEURONS)).cuda()

net_optimizer = optim.Adam(dense_net.parameters(),lr=LEARNING_RATE)

loss_func = nn.MSELoss()

if not os.path.exists('./model'):
    os.mkdir('./model')

try:
    dense_net = torch.load(os.path.join('./model',MODEL_NAME))
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")



for epoch in range(EPOCHS):
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
            print('Ground Truth: {}'.format(ground_truth))
            print('Predictions {}'.format(predictions))
            print('Loss: {}'.format(loss))
            torch.save(dense_net,os.path.join('./model',MODEL_NAME))
        





    
