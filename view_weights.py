import pickle
import torch
import numpy as np
np.set_printoptions(threshold=np.nan)


MODEL_NAME = './model/forward_model_large_design.pkl'


try:
    model = torch.load(MODEL_NAME)
    print('Model Loaded')
    for param in model.parameters():
        print(param.data)
except:
    print('Model not found')


