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

from helper_functions import *

model = import_model()
x, y = model.x, model.y
title = model.title
cnn = model.cnn

# Tensorboard and saving variables
with tf.name_scope(title):
    with tf.name_scope("Model"):
        prediction = tf.nn.softmax(cnn(x))
    saver = tf.train.Saver()

# Session
def writer():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, PATH + "weights/%s.ckpt" % title)

        ##########################################################################
        print "Testing"
        with h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'w') as hfile:
            dset_pred = hfile.create_dataset('all_test_predictions', shape=(NUM_GOOD_TEST_EVENTS_3, 3), dtype='float')
            i = 0
            for events, labels in model.batches(150, test=True):
                feed_dict = {x: events, y: labels}
                p = sess.run(prediction, feed_dict=feed_dict)
                dset_pred[i: i + 150] = p

                print i, NUM_GOOD_TEST_EVENTS_3
                i += 150
            

    ##########################################################################
f = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
q = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')

j = NUM_GOOD_TRAIN_EVENTS_3
good, fout = 0, 0
for i in range(len(f['all_test_predictions'])):
    if np.argmax(f['all_test_predictions'][i]) == np.argmax(q['all_labels'][j]):
        good += 1
    else:
        fout += 1
    j += 1

print good / float(good + fout)
