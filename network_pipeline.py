"""
    network_pipeline.py
    Maarten Post 2018
"""

import tensorflow as tf
import numpy as np
import h5py

from helper_functions import *
from models.tf_help import weight, bias, conv3d, maxpool3d

title = 'pipeline'

train_data_file = PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta.hdf5'
val_data_file   = PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta_val.hdf5'
test_data_file  = PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta_test.hdf5'

def event_label_train_gen():
    with h5py.File(train_data_file, 'r') as hf:
        for i in range(NUM_TRAIN_EVENTS):
            label = hf['all_labels'][i]

            bins = hf['all_bins'][i]
            tots = hf['all_tots'][i]
            event = np.zeros((NUM_MINI_TIMESLICES, 13, 13, 18, 3))
            event[tuple(bins)] = tots
            
            yield event, label

def event_label_val_gen():
    with h5py.File(val_data_file, 'r') as hf:
        for i in range(NUM_TRAIN_EVENTS):
            label = hf['all_labels'][i]

            bins = hf['all_bins'][i]
            tots = hf['all_tots'][i]
            event = np.zeros((NUM_MINI_TIMESLICES, 13, 13, 18, 3))
            event[tuple(bins)] = tots
            
            yield event, label

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
    
    conv1 = tf.contrib.layers.batch_norm(conv1)

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv1 = tf.contrib.layers.batch_norm(conv1)

    conv2 = maxpool3d(conv2)

    conv2 = tf.contrib.layers.batch_norm(conv2)

    fc = tf.reshape(conv2, [-1, 11025])
    
    fc = tf.nn.relu(
        tf.matmul(fc, weights["l3"]) + biases["l3"])

    fc = tf.nn.dropout(fc, keep_prob)

    conv1 = tf.contrib.layers.batch_norm(fc)

    fc = tf.nn.relu(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    return fc

def km3nnet(evt):
    mini_timeslices = tf.unstack(evt, num_mini_timeslices, 1)

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

saver = tf.train.Saver()
# TRAINING 
train_data_set = tf.data.Dataset.from_generator(event_label_train_gen, (tf.float32, tf.int32)).batch(15)
event,  label  = train_data_set.make_one_shot_iterator().get_next()

output = km3nnet(event)
prediction = tf.nn.softmax(output)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.03, global_step, NUM_TRAIN_EVENTS, 0.97, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.7)
train_op = optimizer.minimize(cost, global_step=global_step)

# TESTING OR VALIDATING
validation_data_set  = tf.data.Dataset.from_generator(event_label_val_gen, (tf.float32, tf.int32)).batch(600)
vevent, vlabel = validation_data_set.make_one_shot_iterator().get_next()

voutput = km3nnet(vevent)
prediction = tf.nn.softmax(voutput)
vcost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=vlabel))
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

with tf.Session() as sess:
    sess.run(tf.gloabel_variables_initializer())
    
    while True:
        loss, _  = sess.run([cost, train_op])

        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title)
