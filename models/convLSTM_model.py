from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import h5py

from tf_help import conv3d, maxpool3d, weight, bias
from helper_functions import NUM_CLASSES

title = 'convlstm'
num_mini_timeslices = 50 

x = tf.placeholder(tf.float32, [None, num_mini_timeslices, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

def lstm_cell():
    return tf.contrib.rnn.ConvLSTMCell(conv_ndims=3, input_shape=[13, 13, 18, 3], output_channels=10, kernel_shape=[4, 4, 4])

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices
    mini_timeslices = tf.unstack(x, num_mini_timeslices, 1)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])
    

    outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm, mini_timeslices, dtype=tf.float32)
    output = tf.reshape(outputs[-1], [-1, 13 * 13 * 18 * 10 ])

    W = weight([13 * 13 * 18 * 10, 100])
    b = bias(100)
    z = tf.matmul(output, W) + b

    K = weight([100, NUM_CLASSES])
    k = bias(NUM_CLASSES)

    output = tf.matmul(z, K) + k
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
