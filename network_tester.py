###################################
# Maarten Post
# network tester
###################################

import tensorflow as tf
import numpy as np
import h5py
import sys
import importlib
import matplotlib.pyplot as plt
from time import time
from helper_functions import *

model = import_model()
x, y = model.x, model.y
title = model.title
cnn = model.cnn

# Tensorboard and saving variables
with tf.name_scope(title):
    with tf.name_scope("Model"):
        output = cnn(x)
        prediction = tf.nn.softmax(output)
    saver = tf.train.Saver()

# Session
def writer():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, PATH + "weights/%s.ckpt" % title)

        ##########################################################################
        print "Testing"
        with h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'w') as hfile:
            dset_pred = hfile.create_dataset('all_test_predictions', shape=(NUM_GOOD_TEST_EVENTS_3, model.NUM_CLASSES), dtype='float')
            i = 0
            batch_size = 10
            for events, labels in model.batches(batch_size, test=True):
                ts = time()
                feed_dict = {x: events, y: labels}
                #p = sess.run(prediction, feed_dict=feed_dict)
                p = sess.run(output, feed_dict=feed_dict)

                dset_pred[i: i + batch_size] = p

                print i, NUM_GOOD_TEST_EVENTS_3
                i += batch_size
                print '\tprediction', (time() - ts) / batch_size

writer()
