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


parser = argparse.ArgumentParser(description='This script will train a network for the forward pass of the data')
parser.add_argument('--data_dir',default='./data/all_design.pkl',help='path to directory containing the layer designs and outputs')
parser.add_argument('--model_dir',default='./model',help='Directory to save and load the model')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--G_model_name',default='Generator_model.pkl')
parser.add_argument('--D_model_name',default='Discriminator_model.pkl')
parser.add_argument('--hidden_neurons',default=512,type=int)
parser.add_argument('--learning_rate',default=0.0005,type=float)
parser.add_argument('--epochs',default=10000,type=int)
parser.add_argument('--z_dim',default=100,type=int,help='size of the latent distribuation')
options = parser.parse_args()
print(options)


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    print('Manual GPU loading successful')  
    print('Number CUDA Devices:', torch.cuda.device_count())
except:
    print('Manual GPU loading failed')
    pass

    


layer_dataset = Layer_Dataset(options.data_dir,mode='gan')
data_loader = DataLoader(layer_dataset, batch_size=options.batch_size,shuffle=True,drop_last=True,num_workers=2)

G = (Generator(in_dim=options.z_dim,out_dim=12,num_units=options.hidden_neurons)).double().cuda()
D = (Discriminator(in_dim=12,out_dim=1,num_units=options.hidden_neurons*2)).double().cuda()


try:
    G = torch.load('./model/Generator.pkl')
    D = torch.load('./model/Discriminator.pkl')
    print('Model loaded')
except:
    print('Model not loaded')
    pass


loss_func = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(),lr=options.learning_rate)
D_optimizer = optim.Adam(D.parameters(),lr=options.learning_rate)

for epoch in tqdm(range(options.epochs)):
    for index,sample in enumerate(data_loader):

        full_data = sample['full_data']
        # 1. Train discriminator
        
        # 1.1 Train discriminator on real data
        
        D.zero_grad()
        d_real_data = Variable(full_data).cuda().double() # Need to cast float to double to avoid nan
        d_real_decision = D(d_real_data)
        d_real_label = Variable(torch.ones(options.batch_size,1)).double().cuda() # Create true labels that equal the size of the batch
        d_real_loss = loss_func(d_real_decision,d_real_label)

        d_real_loss.backward() # Calculate gradients

        # 1.2 Train discriminator on fake data 
        g_input_z = torch.randn(options.batch_size, options.z_dim).view(-1, options.z_dim) # Create random data as inputs for generator
        g_input_z = Variable(g_input_z).double().cuda()
        d_fake_label = Variable(torch.zeros(options.batch_size)).double().cuda() # Create labels for the fake data that equal the size of the batch
        fake_data = G(g_input_z).detach()#Avoid training G here
        d_fake_decision = D(fake_data)
        d_fake_loss = loss_func(d_fake_decision,d_fake_label)

        d_fake_loss.backward()
        D_optimizer.step()


        # 2. Train generator

        G.zero_grad()
        g_input_z = torch.randn(options.batch_size, options.z_dim).view(-1, options.z_dim) # Create random data as inputs for generator
        g_input_z = Variable(g_input_z).double().cuda()

        fake_data = G(g_input_z)
        d_fake_decision = D(fake_data)

        g_loss = loss_func(d_fake_decision,d_real_label) # fake decision should be close to real label

        g_loss.backward()
        G_optimizer.step()

        if index % 100 == 0:
            print('\n')
            print('G loss {}'.format(g_loss))
            print('D real loss {}'.format(d_real_loss))
            print('D fake loss {}'.format(d_fake_loss))

            print('True data {}'.format(d_real_data))
            print('Generated data {}'.format(fake_data))

            torch.save(D,'./model/Discriminator.pkl')
            torch.save(G, './model/Generator.pkl')












