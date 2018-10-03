###################################
# Maarten Post
# network tester
###################################
""" This program takes a trained NN and runs it over the test set.
    Then it writes all the output to file. The output is the softmax threevector
    for each evetn"""
import tensorflow as tf
import numpy as np
import h5py
import sys
import importlib
import matplotlib.pyplot as plt
from time import time
from helper_functions import *

model = import_model()

# Tensorboard and saving variables
saver = tf.train.Saver()

# Session
def writer():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, PATH + "weights/%s.ckpt" % model.title)

        ##########################################################################
        print "Testing"
        with h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (model.title, model.title), 'w') as hfile:
            dset_pred = hfile.create_dataset('all_test_predictions', shape=(NUM_GOOD_TEST_EVENTS_3, model.NUM_CLASSES), dtype='float')
            i = 0
            batch_size = 30 
            for events, labels in model.batches(batch_size, test=True):

                ts = time()

                feed_dict = {model.x: events, model.y: labels, model.keep_prob = 1.}
                p = sess.run(model.prediction, feed_dict=feed_dict)

                dset_pred[i: i + batch_size] = p

                print i, NUM_TEST_EVENTS
                i += batch_size
                print '\tprediction', (time() - ts) / batch_size

writer()
