# coding=utf-8
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
data = pd.read_pickle('all_design.pkl')
data = data.dropna()
data = data.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
#data=(data-data.mean())/data.std() # Normalize the data 

# For all_design.pkl forward training output, select labels R, T, A
output_labels = ['R','T','A']

# Set up a diff function
diff = lambda l1,l2: [x for x in l1 if x not in l2]



# Grab the training labels
input_labels = diff(list(data.columns),output_labels)

#Switch up the input output
output_labels = input_labels
input_labels = ['R','T','A']



#For tensorboard visualization
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

def dense_forward(inputs, activation):
    # First layer
    dense1_size = 3200
    dense1 = tf.layers.dense(inputs, dense1_size, activation=activation)
    # Second layer
    dense2_size = 2500
    dense2 = tf.layers.dense(dense1, dense2_size, activation=activation)
    # Third layer
    dense3_size = 1000
    dense3 = tf.layers.dense(dense2, dense3_size, activation=activation)
    # Fourth layer
    dense4_size = 800
    dense4 = tf.layers.dense(dense3, dense4_size, activation=activation)
    # Fifth layer
    dense5_size = 600
    dense5 = tf.layers.dense(dense4, dense5_size, activation=activation)
    # Sixth layer
    dense6_size = 400
    dense6 = tf.layers.dense(dense5, dense6_size, activation=activation)
    # Output layer
    output = tf.layers.dense(dense6, len(output_labels), activation=activation)
    
    return output

n_epochs = 30
batch_size = 256
lr = 0.001

layer_inputs = tf.placeholder(tf.float32, (None, len(input_labels)))
layer_labels = tf.placeholder(tf.float32, (None, len(output_labels)))

layer_outputs = dense_forward(layer_inputs, tf.nn.relu)
loss = tf.reduce_mean(tf.square(layer_outputs - layer_labels))  # claculate the mean square error loss
abs_diff = tf.reduce_mean(tf.abs((layer_outputs - layer_labels)))
avg_pdiff = tf.reduce_mean(2.*(layer_outputs - layer_labels)/(layer_outputs + layer_labels))

all_avg_pdiff = tf.reduce_mean(2.*(layer_outputs - layer_labels)/(layer_outputs + layer_labels),axis= 0)

train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

avg_pdiff_arr = []

r_loss_arr = []
t_loss_arr = []
a_loss_arr = []
r_loss = scalar_summary("R_loss", all_avg_pdiff[0])
t_loss = scalar_summary("D_loss",all_avg_pdiff[1])
a_loss = scalar_summary("A_Loss",all_avg_pdiff[2])
#train_summary = histogram_summary('Training Summary',train_op)
total_loss = merge_summary([r_loss,t_loss,a_loss])

SummaryWriter("./logs_inverse_model", sess.graph)
# Create a saver
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.5)

# Now sort into train, validation, test sets
validation_size = int(np.floor(len(data)/10))
test_size = int(np.floor(len(data)/10))
train_size = len(data) - validation_size - test_size

validation = data.ix[0:(validation_size-1)]
test = data.ix[validation_size:(validation_size+test_size-1)]
train = data.ix[(validation_size+test_size):(validation_size+test_size+train_size)]

# # Now sort into train, validation, test sets
# data = data.sample(frac =1).reset_index(drop = True)
# validation_size = int(np.floor(len(data)/10))
# test_size = int(np.floor(len(data)/10))
# train_size = len(data) - validation_size - test_size

# test= data.ix[0:(test_size-1)]
# data = data.ix[test_size:]

with tf.Session() as sess:
    init.run()
    
    for epoch in range(n_epochs):
        train = train.sample(frac=1).reset_index(drop=True) # Shuffle the data for training purpose
        print("Size of Training set")
        print(len(train))
        print("Size of Validation set")
        print(len(validation))
        
        for iteration in range((train_size//10) // batch_size):
            feed_dict = {layer_inputs: train[input_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)], layer_labels: train[output_labels].as_matrix()[(iteration*batch_size):(((iteration+1)*batch_size)-1)]}
            sess.run(train_op, feed_dict = feed_dict)
            avg_pdiff_val = avg_pdiff.eval(feed_dict=feed_dict)
            abs_diff_val = abs_diff.eval(feed_dict = feed_dict)
            loss_val = loss.eval(feed_dict)
            print("Loss: {}".format(str(loss_val)))

            #print('Step'+str(iteration))
            
        epoch_feed_dict = {layer_inputs: validation[input_labels].as_matrix(), layer_labels: validation[output_labels].as_matrix()}
        print("Epoch: " + str(epoch))
        avg_pdiff_val = avg_pdiff.eval(feed_dict=epoch_feed_dict)
        print("Loss: {}".format(str(avg_pdiff_val)))
        avg_pdiff_arr.append(avg_pdiff_val)
        #print("Loss: {}".format(str(loss_val)))
        all_avg_pdiff_val = all_avg_pdiff.eval(feed_dict=epoch_feed_dict)
        r_loss_val = all_avg_pdiff_val[0]
        t_loss_val = all_avg_pdiff_val[1]
        a_loss_val = all_avg_pdiff_val[2]
        r_loss_arr.append(r_loss_val)
        t_loss_arr.append(t_loss_val)
        a_loss_arr.append(a_loss_val)
        
        print("R Loss: {}".format(str(r_loss_val)))
        print("T Loss: {}".format(str(t_loss_val)))
        print("A Loss: {}".format(str(a_loss_val)))
        saver.save(sess, './layer_model')

    plt.figure()
    plt.plot(range(n_epochs),avg_pdiff_arr, 'k', range(n_epochs), r_loss_arr, 'r', range(n_epochs), t_loss_arr, 'g', range(n_epochs), a_loss_arr, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Mean', 'R', 'T', 'A'])
    plt.show()


print(str(avg_pdiff_arr))
print(str(r_loss_arr))
print(str(t_loss_arr))
print(str(a_loss_arr))