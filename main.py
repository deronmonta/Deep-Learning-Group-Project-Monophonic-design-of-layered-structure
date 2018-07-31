from dataset.layer_dataset import *

DATA_DIR = './data/all_design.pkl'


layer_dataset = Layer_Dataset(DATA_DIR) 


for i,sample in enumerate(layer_dataset):
    print(sample)