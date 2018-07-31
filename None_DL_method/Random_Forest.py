from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
plt.switch_backend('agg')
# to make this notebook's output stable across runs
# def reset_graph(seed=42):
#     tf.reset_default_graph()
#     tf.set_random_seed(seed)
#     np.random.seed(seed)

#reset_graph()

# Set up logging
tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.InteractiveSession()

# Load in the training data
# all_design.pkl
#
# R - total reflected light (normalized)
# T - transmission (normalized)
# A - absorption (normalized)
# d1-d8 - thickness of layer (Si, SiO2, alternating) (nm, I assume)
# lambda - wavelength of input light
data = pd.read_pickle(r'C:\Users\FurryMonster Yang\all_design.pkl')
data = data.dropna()
data = data.sample(frac=0.001).reset_index(drop=True) # Shuffle the data for training purpose
#data=(data-data.mean())/data.std() # Normalize the data 

# For all_design.pkl forward training output, select labels R, T, A
output_labels = ['R','T','A']

# Set up a diff function
diff = lambda l1,l2: [x for x in l1 if x not in l2]

# Grab the training labels
input_labels = diff(list(data.columns),output_labels)

# Now sort into train, validation, test sets
validation_size = int(np.floor(len(data)/10))
test_size = int(np.floor(len(data)/10))
train_size = len(data) - validation_size - test_size

data =  data.sample(frac=1).reset_index(drop=True)
validation = data.ix[0:(validation_size-1)]
test = data.ix[validation_size:(validation_size+test_size-1)]
train = data.ix[(validation_size+test_size):(validation_size+test_size+train_size)]


#train[input_labels], train[output_labels] = make_regression(n_features=9, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=20, random_state=0)
regr.fit(train[input_labels], train[output_labels])

#print(regr.feature_importances_)
pred_results = regr.predict(test[input_labels])
print("Test Results:")
print(pred_results)
print("True Label")
print(test[output_labels].as_matrix())

mse = mean_squared_error(test[output_labels].as_matrix(), pred_results)
print('Mean Square Error:')
print(mse)