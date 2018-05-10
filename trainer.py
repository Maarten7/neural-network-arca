###################################
# Maarten Post
# Slice detector
###################################
""" This neural network should take the detector slices and with shared weights
map them to an array or matrix and than this one goes to the output"""
import tensorflow as tf
import numpy as np
import datetime
import h5py
import sys
import importlib
from time import time
from helper_functions import * 

model = sys.argv[1]
model = importlib.import_module(model)
title = model.title

# Input; inp is vector of 30*3 events containing 13 slices
# of detector.
debug = eval(sys.argv[2])
num_epochs = 1000 if not debug else 2
f = h5py.File(PATH + 'data/hdf5_files/bg_file_%s.hdf5' % title, 'r')
#################################################################
# Neural network.
# shared weights should be used on all the 13 slices
# and create 13 (scalars or vectors) to get vector or matrix
# here another CNN will go over and then fully connected
#################################################################

# Loss & Training
# Compute cross entropy as loss function
with tf.name_scope(title):
    with tf.name_scope("Model"):
        prediction = model.cnn(model.x)
    with tf.name_scope("Xentropy"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=model.y))
    # Train network with AdamOptimizer
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    # Compute the accuracy
    with tf.name_scope("Test"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    saver = tf.train.Saver()

def test_model(sess):
    acc = [] 
    ecc, mcc, k40 = 0, 0, 0
    
    loss = 0
    for root_file, _ in root_files(train=False, test=True):
        events, labels = f[root_file].value, f[root_file + 'labels'].value
        num_ecc_events, num_mcc_events, num_k40_events =  np.sum(labels, axis=0)
        ecc += num_ecc_events 
        mcc += num_mcc_events 
        k40 += num_k40_events
        
        feed_dict = {model.x: events, model.y: labels}
        p, a, c = sess.run([prediction, accuracy, cost], feed_dict=feed_dict)
        acc.append(a * len(labels))
        loss += c
    
    acc = sum(np.array(acc)) / float(ecc+mcc+k40)
    open(PATH + 'data/results/%s/_acc_test/' % title + timestamp() + str(acc), 'w')
    open(PATH + 'data/results/%s/_cost_test/' % title + timestamp() + str(loss / NUM_TEST_EVENTS), 'w')
    return acc


def train_model(sess, test=True):
    print 'Start training'
    for epoch in range(num_epochs):
        print "epoch", epoch, "\tbatch"
        acc = [] 
        ecc, mcc, k40 = 0, 0, 0

        epoch_loss = 0
        #######################################################################
        for i, (root_file, _) in enumerate(root_files(debug=debug)):
            print '\t', i
            events, labels = f[root_file].value, f[root_file + 'labels'].value
            num_ecc_events, num_mcc_events, num_k40_events =  np.sum(labels, axis=0)
            ecc += num_ecc_events 
            mcc += num_mcc_events 
            k40 += num_k40_events

            _, a, c = sess.run([optimizer, accuracy, cost], feed_dict={model.x: events, model.y: labels})
            acc.append(a * len(labels))
            epoch_loss += c
        
        acc = sum(np.array(acc)) / float(ecc+mcc+k40)
        print "cost", epoch_loss
        open(PATH + 'data/results/%s/_acc_train/' % title + timestamp() + str(acc), 'w')
        open(PATH + 'data/results/%s/_cost_train/' % title + timestamp() + str(epoch_loss / NUM_TRAIN_EVENTS), 'w')
        if test and epoch % 10 == 0: print 'Accuracy', test_model(sess)
        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title)
        ########################################################################
    return epoch_loss


def main():
    # Session
    print 'Start session'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, PATH + "weights/%s.ckpt" % title)
        except:
            print 'Initalize variables'
            sess.run(tf.global_variables_initializer())
        
        train_model(sess, test=True)
 


if __name__ == "__main__":
    t_start = time()
    a = main()
    t_end = time()
    print 'runtime', t_end - t_start
        
# dropout
# learning rate
# num conv layers
# num fc layers
# size of convolutions
# - weigts and biases
# activation 
# optimiers
