###################################
# trainer.py
# Maarten Post
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

# Importing model to train
model = sys.argv[1].replace('/', '.')[:-3]
model = importlib.import_module(model)
title = model.title

# Settings for debug session
debug = eval(sys.argv[2])
num_epochs = 1000 if not debug else 2
num_events = NUM_DEBUG_EVENTS if debug else NUM_TRAIN_EVENTS

# output of data_writer
f = h5py.File(PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title, 'r')

# Loss & Training
# Compute cross entropy as loss function
with tf.name_scope(title):
    # Imported cnn model from model 
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
    # Saver objects enables to save trained weights  
    saver = tf.train.Saver()

def test_model(sess):
    """ test model on test data set
        input sess: a tensor flow session"""
    print "Testing..."
    acc, loss = [], 0
    for root_file, _ in root_files(train=False, test=True):
        # temporal    
        tots, bins, labels = f[root_file + 'tots'], f[root_file + 'bins'], f[root_file + 'labels']
        for j in range(len(labels)):
            t = tots[j] 
            b = tuple(bins[j])
            
            event = np.emtpy((bins[0,-1], 13, 13, 18, 31))
            event[b] = t

            feed_dict = {model.x: event, model.y: labels[i]} 
            a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
            acc.append(a)
            loss += c
            # for other
            if 1 == 2:
                events = f[root_file]
                feed_dict = {model.x: events.value, model.y: labels.value}
                a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
                acc.append(a * len(labels))
                loss += c
    
    acc = sum(np.array(acc)) / float(NUM_TEST_EVENTS)
    save_output(acc, loss, test=True)
    return acc


def train_model(sess, test=True):
    """ trains model,
        input sess, a tensorflow session.
        input test, boolean, default True, if True the accuracy and cost
                    of test set are calculated"""
    print 'Start training...'
    num_events = NUM_DEBUG_EVENTS if debug else NUM_TRAIN_EVENTS
    batch_size = 10
    for epoch in range(num_epochs):
        print "epoch", epoch 

        acc, epoch_loss = [], 0 

        #######################################################################
        for i, (root_file, _) in enumerate(root_files(debug=debug)):
            # for temporal
            tots, bins, labels = f[root_file + 'tots'], f[root_file + 'bins'], f[root_file + 'labels']
            for j in range(len(labels)):
                t = tots[j] 
                b = tuple(bins[j])
                
                event = np.emtpy((bins[0,-1], 13, 13, 18, 31))
                event[b] = t

                feed_dict = {model.x: event, model.y: labels[i]} 
                _, a, c = sess.run([optimizer, accuracy, cost], feed_dict=feed_dict)
                acc.append(a)
                epoch_loss += c
            # for other
            if 1 == 2:
                events = f[root_file]
                feed_dict = {model.x: events.value, model.y: labels.value}
                _, a, c = sess.run([optimizer, accuracy, cost], feed_dict=feed_dict)
                acc.append(a * len(labels))
                epoch_loss += c

       
        acc = sum(np.array(acc)) / float(num_events)

        save_output(acc, epoch_loss)

        if test and epoch % 10 == 0: print 'Accuracy', test_model(sess)
        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title)
        ########################################################################
    return epoch_loss


def main():
    # Tensorflow Session; here the cnn model is accualy trained.
    print 'Start session'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:

        # Is model is already trained restores the weights
        try:
            saver.restore(sess, PATH + "weights/%s.ckpt" % title)
        except:
            print 'Initalize variables'
            sess.run(tf.global_variables_initializer())

        # TRAINING OF MODEL
        train_model(sess, test=False)
 


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
