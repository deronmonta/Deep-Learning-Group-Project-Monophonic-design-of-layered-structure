import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import dataset
plt.switch_backend('agg')


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)




X = tf.placeholder(tf.float32, shape=[None, img_size*img_size])#None is the mini batch size
#Do batch normalization here
# epsilon = 1e-3
# scale2 = tf.Variable(tf.ones([img_size*img_size]))
# beta2 = tf.Variable(tf.zeros([img_size*img_size]))
# batch_mean2, batch_var2 = tf.nn.moments(X,[0])

# X = tf.nn.batch_normalization(X,batch_mean2,batch_var2,beta2,scale2,epsilon)

#Variables for Encoder
E_W1 = tf.Variable(xavier_init([-1, 512]))
E_b1 = tf.Variable(tf.zeros(shape=[512]))

E_W2 = tf.Variable(xavier_init([512, 256]))
E_b2 = tf.Variable(tf.zeros(shape=[1]))

E_W3 = tf.Variable(xavier_init([256, 3]))
E_b3 = tf.Variable(tf.zeros(shape=[3]))

D_W1 = tf.Variable(xavier_init([3, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))



theta_D = [E_W1, E_W2, E_b1, E_b2, E_W3, E_b3]



def encoder(x):
    E_h1 = tf.nn.relu(tf.matmul(x, E_W1) + E_b1)
    E_h2 = tf.nn.relu(tf.matmul(E_h1, E_b1) + E_b2)
    E_output = tf.matmul(E_h2, E_W3) + E_b3
    y_true = tf.placeholder(tf.float32, [None, 3])

    return E_output

def decoder()

loss1 = tf.reduce_mean(tf.square(E_output-y_true))
loss2 = 

def input_fn(pred = False, batch_size = 256):
        
    if pred == False:
        indices = np.random.choice(range(len(train)), batch_size)#select random indices for minibatch
        feature_cols = {k: tf.constant(train.iloc[indices][k].values) for k in FEATURES}
        labels = tf.constant(train.iloc[indices][LABEL].values)
        
        return feature_cols, labels
trainer = tf.train.AdamOptimizer().minimize(loss1+loss2)


sess = tf.InteractiveSession(); # start the session
tf.global_variables_initializer().run();

batch_size = 256
for j in range(2000):
    for i in range( int((TotalSampleSize-MiniTestSampleSize)/200) ): # lets take 200 samples at time
        indices = np.random.choice(range(len(train)), batch_size)#select random indices for minibatch
        feature_cols = {k: tf.constant(train.iloc[indices][k].values) for k in FEATURES}
        labels = tf.constant(train.iloc[indices][LABEL].values)
        sess.run(trainer, feed_dict={x: feature_cols, y_: labels})
