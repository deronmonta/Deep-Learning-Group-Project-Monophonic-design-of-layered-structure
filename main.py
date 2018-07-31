from torch.utils.data import Dataset, DataLoader

from dataset.layer_dataset import *
from nets.nets import *


#Hyperparameters
DATA_DIR = './data/all_design.pkl'
BATCH_SIZE = 16
NUM_NEURONS = 128




layer_dataset = Layer_Dataset(DATA_DIR) 
data_loader = DataLoader(layer_dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

dense_net = Dense_Net(in_dim=8,out_dim=4,num_neurons=NUM_NEURONS)



for i,sample in enumerate(layer_dataset):
    layer_thickness = sample['layer_thickness'].cuda()
    inputs = sample['Lambda_RTA'].cuda()

    prediction = dense_net(inputs)

    
