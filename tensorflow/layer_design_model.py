# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

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
data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
data=(data-data.mean())/data.std() # Normalize the data 

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

validation = data.iloc[0:(validation_size-1)]
test = data.iloc[validation_size:(validation_size+test_size-1)]
train = data.iloc[(validation_size+test_size):(validation_size+test_size+train_size)]

def dense_forward(inputs, activation):
    # First layer
    dense1_size = 500
    dense1 = tf.layers.dense(inputs, dense1_size, activation=activation)
    # Second layer
    dense2_size = 400
    dense2 = tf.layers.dense(dense1, dense2_size, activation=activation)
    # Third layer
    dense3_size = 200
    dense3 = tf.layers.dense(dense2, dense3_size, activation=activation)
    # Fourth layer
    dense4_size = 100
    dense4 = tf.layers.dense(dense3, dense4_size, activation=activation)
    # Fifth layer
    dense5_size = 50
    dense5 = tf.layers.dense(dense4, dense5_size, activation=activation)
    # Sixth layer
    dense6_size = 25
    dense6 = tf.layers.dense(dense5, dense6_size, activation=activation)
    # Output layer
    output = tf.layers.dense(dense6, len(output_labels), activation=activation)
    
    return output

n_epochs = 100
batch_size = 256
lr = 0.001

layer_inputs = tf.placeholder(tf.float32, (None, len(input_labels)))
layer_labels = tf.placeholder(tf.float32, (None, len(output_labels)))

layer_outputs = dense_forward(layer_inputs, tf.nn.relu)
loss = tf.reduce_mean(tf.square(layer_outputs - layer_labels))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    
    for epoch in range(n_epochs):
        test = test.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
        for iteration in range(train_size // batch_size):
            feed_dict = {layer_inputs: test[input_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)], layer_labels: test[output_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)]}
            sess.run(train_op, feed_dict = feed_dict)
        print('Iter: {}'.format(epoch))
        print("Loss: {}".format(str(loss.eval(feed_dict={layer_inputs: validation[input_labels].as_matrix(), layer_labels: validation[output_labels].as_matrix()}))))
