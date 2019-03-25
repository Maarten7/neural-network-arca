from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

from tf_help import conv3d, maxpool3d, weight, bias
from helper_functions import EVT_TYPES, NUM_CLASSES, PATH, NUM_TRAIN_EVENTS

title = 'temporal_convLSTM'
num_mini_timeslices = 50

x = tf.placeholder(tf.float32, [None, num_mini_timeslices, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

nodes =   {"l1": 25,
           "l2": 25,
           "l3": 25,
           "l4": 1,
           "l5": 20} 
           
weights = {"l1": weight([4, 4, 4, 3, nodes["l1"]]),
           "l2": weight([3, 3, 3,    nodes["l1"], nodes["l2"]]),
           "l3": weight([3, 3, 3,    nodes["l2"], nodes["l3"]]),
           "l4": weight([2, 2, 2,    nodes["l3"], nodes["l4"]])}

biases =  {"l1": bias(nodes["l1"]),
           "l2": bias(nodes["l2"]),
           "l3": bias(nodes["l3"]),
           "l4": bias(nodes["l4"])}

def cnn(mini_timeslice):
    """ input: event tensor numpy shape 1, 13, 13, 18, 3"""
    conv1 = tf.nn.relu(
        conv3d(mini_timeslice, weights["l1"]) + biases["l1"])
    
    conv1 = tf.contrib.layers.batch_norm(conv1)

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv1 = tf.contrib.layers.batch_norm(conv1)
    
    conv2 = maxpool3d(conv2)

    conv2 = tf.contrib.layers.batch_norm(conv2)

    conv3 = tf.nn.relu(
        conv3d(conv2, weights["l3"]) + biases["l3"])

    conv3 = tf.contrib.layers.batch_norm(conv3)
    
    conv3 = maxpool3d(conv3)

    conv4 = tf.nn.relu(
        conv3d(conv3, weights["l4"]) + biases["l4"])

    return conv4 
   
def lstm_cell():
    return tf.contrib.rnn.ConvLSTMCell(conv_ndims=3, input_shape=[4, 4, 5, 1], output_channels=10, kernel_shape=[2, 2, 2])

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices

    mini_timeslices = tf.unstack(x, num_mini_timeslices, 1)

    out_time_bin = []
    for ts in mini_timeslices:
        out_time_bin.append(cnn(ts))

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])

    outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm, out_time_bin, dtype=tf.float32)
    fout = []
    for o in outputs:
        output = tf.reshape(o, [-1, 4 * 4 * 5 * 10 ])

        W = weight([4 * 4 * 5 * 10, NUM_CLASSES])
        b = bias(NUM_CLASSES)
        output = tf.matmul(output, W) + b
        fout.append(tf.nn.softmax(output))

    return output, fout


output, fout = km3nnet(x)
prediction = tf.nn.softmax(output)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.7)
#gvs = optimizer.compute_gradients(cost)
#capped_gvs= [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_op = optimizer.apply_gradients(capped_gvs)
train_op = optimizer.minimize(cost)


correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

if __name__ == "__main__":
    pass
