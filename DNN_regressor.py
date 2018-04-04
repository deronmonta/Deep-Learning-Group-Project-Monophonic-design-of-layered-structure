from __future__ import division
from __future__ import print_function
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
from sklearn.ensemble import IsolationForest
import os
import tensorflow as tf
import warnings 

import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

<<<<<<< HEAD
tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

train = pd.read_pickle('all_design.pkl')#Cotaining all training data
print('Shape of the train data with all features:', train.shape)
print("")
print('Shape of the train data with numerical features:', train.shape)

print("")
print("List of features contained our dataset:",list(train.columns))
#%%
col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('R')
col_train_bis.remove('T')
col_train_bis.remove('A')
mat_train = np.matrix(train.drop(['R','T','A'],axis =1))
#%%
=======
# Test with a simple computation

tf.Session()

with tf.device('/cpu:0'):
    # Creates a graph
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
# If you have gpu you can try this line to compute b with your GPU
#with tf.device('/gpu:0'):    
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# Runs the op.
# Log information
options = tf.RunOptions(output_partition_graphs=True)
metadata = tf.RunMetadata()
c_val = sess.run(c, options=options, run_metadata=metadata)

print(metadata.partition_graphs)

sess.close()




tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

train = pd.read_csv('inputdata.csv')
print('Shape of the train data with all features:', train.shape)
train = train.select_dtypes(exclude=['object'])
print("")
print('Shape of the train data with numerical features:', train.shape)
#train.drop('Id',axis = 1, inplace = True)
train.fillna(0,inplace=True)

test = pd.read_csv('test.csv')
test = test.select_dtypes(exclude=['object'])
ID = test.Id
test.fillna(0,inplace=True)
test.drop('Id',axis = 1, inplace = True)

print("")
print("List of features contained our dataset:",list(train.columns))




clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train.shape[0])

train.head(10)


warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove('output')

mat_train = np.matrix(train)
mat_test  = np.matrix(test)
mat_new = np.matrix(train.drop('output',axis = 1))
mat_y = np.array(train.output).reshape((333,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
#test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

train.head()


>>>>>>> parent of f54f9a8... Data,preprocessing and DNN regressor
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "output"

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict
training_set = train[COLUMNS]
prediction_set = train.output

<<<<<<< HEAD
prediction_set = train[['R','T','A']]
#%% Define the model and the input function
#Model Here

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[800, 400, 200, 100, 50, 25],optimizer = tf.train.AdamOptimizer(learning_rate= 0.1),
                                          label_dimension=3)
#hidden_units=[400, 200, 100, 50, 25, 50]
def input_fn(pred = False, batch_size = 256):
=======
# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
training_set.head()

# Training for submission
training_sub = training_set[col_train]

#Model Begins Here-------------------------------------------------------------------------------

y_test = pd.DataFrame(y_test, columns = [LABEL])
testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
testing_set.head()


tf.logging.set_verbosity(tf.logging.ERROR)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])#,
                                         #optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 ))
                                         


training_set.reset_index(drop = True, inplace =True)

def input_fn(data_set, pred = False):
>>>>>>> parent of f54f9a8... Data,preprocessing and DNN regressor
    
    if pred == False:
        
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        
        return feature_cols, labels

    if pred == True:
        test_set_size = 10
        indices = np.random.choice(range(len(train)), test_set_size)#Chooes the number of minibatch as indices
        feature_cols = {k: tf.constant(train.iloc[:10][k].values) for k in FEATURES}
        
        return feature_cols
<<<<<<< HEAD
#%%Mini Batch training
      
epochs = 10
#batch_size =128
init_op = tf.global_variables_initializer()
#saver = tf.train.Saver()
#with tf.Session() as sess:
for e in range(epochs):
    regress_res = regressor.train(input_fn = input_fn, steps=1000)
    ev = regressor.evaluate(input_fn = input_fn, steps=1)
    #sess.close()
        
    #saver.save(sess2, "/model.ckpt")
    # Shuffle training data
    

#%%Prediction Mode
input_fn_pred = input_fn(pred = True)
with tf.Session() as sess:
    y = regressor.predict(input_fn = lambda: input_fn(pred = True))
    predictions = list(itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))
    sess.close()
=======
    
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# 0.002X in average
loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))
y = regressor.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
predictions = prepro_y.inverse_transform(np.array(predictions).reshape(110,1))
>>>>>>> parent of f54f9a8... Data,preprocessing and DNN regressor
