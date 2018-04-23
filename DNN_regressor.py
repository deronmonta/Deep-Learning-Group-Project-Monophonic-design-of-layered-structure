from __future__ import division
from __future__ import print_function
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import tensorflow as tf
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib



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


COLUMNS = col_train
FEATURES = col_train_bis
LABEL = ['R','T','A']

# Columns for tensorflow
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
train=(train-train.mean())/train.std()#Normalize the data 


#Subset into train and test data
#indices = pd.DataFrame(np.random.randn(train))
msk = np.random.choice(range(len(train)),int(0.2*len(train)))
test = train.iloc[msk]
train = train.drop(msk[:])



regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[500, 400, 200, 100, 50, 25],optimizer = tf.train.AdamOptimizer(learning_rate= 0.001),
                                          label_dimension=3)
#hidden_units=[400, 200, 100, 50, 25, 50]

def input_fn(pred = False, batch_size = 256):
        
    if pred == False:
        indices = np.random.choice(range(len(train)), batch_size)#select random indices for minibatch
        feature_cols = {k: tf.constant(train.iloc[indices][k].values) for k in FEATURES}
        labels = tf.constant(train.iloc[indices][LABEL].values)
        
        return feature_cols, labels

        


#Input function for testing

def input_test():
    test_set_size = 100
    indices = np.random.choice(range(len(test)), test_set_size)
    feature_cols = {k: tf.constant(test.iloc[indices][k].values) for k in FEATURES}
        
    return feature_cols



#%%Mini Batch training

      
epochs = 10
#batch_size =128
init_op = tf.global_variables_initializer()
logs_path = r'D:\Yale_Course\Deep Learning Theory Application\Deep-Learning-Group-Project-Monophonic-design-of-layered-structure\tensor_board_test'
#saver = tf.train.Saver()
with tf.Session() as sess:
    #writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for e in range(epochs):
        regress_res = regressor.train(input_fn = input_fn, steps=1000)
       # ev = regressor.evaluate(input_fn = input_test, steps=1)#Evaluate the model every epoch
    #save_path = saver.save(sess, logs_path)
    

#%%Prediction Mode
input_fn_pred = input_fn(pred = True)
with tf.Session() as sess:
    y = regressor.predict(input_fn = lambda: input_fn(pred = True))
    predictions = list(itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))
    sess.close()


#regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)
#ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
#
## 0.002X in average
#loss_score1 = ev["loss"]
#print("Final Loss on the testing set: {0:f}".format(loss_score1))
#y = regressor.predict(input_fn=lambda: input_fn(testing_set))
#predictions = list(itertools.islice(y, testing_set.shape[0]))
#predictions = prepro_y.inverse_transform(np.array(predictions).reshape(110,1))
