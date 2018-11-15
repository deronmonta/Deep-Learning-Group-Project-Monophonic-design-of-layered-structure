import pickle
import torch
import numpy as np
import os
np.set_printoptions(threshold=np.nan)


MODEL_NAME = './model/forward_model_large_design.pkl'
EXPORT_WEIGHTS = 'export_weights'
EXTENSION = '.csv'
FILENAMES = ['weights1','bias1',
            'weights2','bias2',
            'weights3','bias3',
            'weights4','bias4',
            'weights5','bias5']
SAVE_DIR = 'weights_results'

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

try:
    model = torch.load(MODEL_NAME,map_location= lambda storage, loc:storage)
    print('Model Loaded')
    for index,param in enumerate(model.parameters()):
        print((param.shape))
        print('Saving {}'.format(FILENAMES[index]))
        param_np = param.data.numpy()
        np.savetxt(os.path.join(SAVE_DIR,FILENAMES[index]+EXTENSION), param_np, delimiter=',')

except:
    print('Model not found')


    