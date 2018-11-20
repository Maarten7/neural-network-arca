from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

from tf_help import conv3d, maxpool3d, weight, bias
from helper_functions import EVT_TYPES, NUM_CLASSES

title = 'temporal'
num_mini_timeslices = 50

x = tf.placeholder(tf.float32, [None, num_mini_timeslices, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

nodes =   {"l1": 25,
           "l2": 25,
           "l3": 80,
           "l4": 40,
           "l5": 20} 
           
weights = {"l1": weight([4, 4, 4, 3, nodes["l1"]]),
           "l2": weight([3, 3, 3, nodes["l1"], nodes["l2"]]),
           "l3": weight([11025, nodes["l3"]]),
           "l4": weight([nodes["l3"], nodes["l4"]])}

biases =  {"l1": bias(nodes["l1"]),
           "l2": bias(nodes["l2"]),
           "l3": bias(nodes["l3"]),
           "l4": bias(nodes["l4"])}

def cnn(mini_timeslice):
    """ input: event tensor numpy shape 1, 13, 13, 18, 3"""
    conv1 = tf.nn.relu(
        conv3d(mini_timeslice, weights["l1"]) + biases["l1"])

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv2 = maxpool3d(conv2)

    elements = np.prod(conv2._shape_as_list()[1:])

    fc = tf.reshape(conv2, [-1, elements])
    
    fc = tf.nn.relu(
        tf.matmul(fc, weights["l3"]) + biases["l3"])

    fc = tf.nn.dropout(fc, keep_prob)

    fc = tf.nn.relu(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    return fc

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices
    mini_timeslices = tf.unstack(x, num_mini_timeslices, 1)
    out_time_bin = []
    for ts in mini_timeslices:
        out_time_bin.append(cnn(ts))
    c = tf.concat(out_time_bin, 1)
    c = tf.reshape(c, [-1, num_mini_timeslices, nodes["l4"]])
    c = tf.unstack(c, num_mini_timeslices, 1)

    lstm_layer = tf.contrib.rnn.BasicLSTMCell(nodes["l5"], forget_bias=1.)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, c, dtype=tf.float32)

    output = tf.matmul(outputs[-1], weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)

    return output 

output = km3nnet(x)
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
