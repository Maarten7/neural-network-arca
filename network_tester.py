###################################
# Maarten Post
###################################

import tensorflow as tf
import numpy as np
import h5py
import sys
import importlib
import matplotlib.pyplot as plt
from time import time
from helper_functions import *

model = sys.argv[1].replace('/', '.')[:-3]
model = importlib.import_module(model)
x, y = model.x, model.y
title = model.title
cnn = model.cnn

# Train data
f = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')
# Write file
z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'w')

# Tensorboard and saving variables
output = cnn(x)
prediction = tf.nn.softmax(output)
saver = tf.train.Saver()

# Session
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config) as sess:
    saver.restore(sess, PATH + "weights/%s.ckpt" % title)

    ##########################################################################
    print "Testing"
    with h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'w') as z:
        dset_p = z.create_dataset('test_predictions', shape=(NUM_GOOD_TEST_EVENTS_3, 3), dtype='float')
        t0 = time()
        for i in range(NUM_GOOD_TRAIN_EVENTS_3, NUM_GOOD_EVENTS_3, 100):
            events, labels = f['all_events'][i: i + 100], f['all_labels'][i: i + 100]
            feed_dict = {x: events, y: labels}
            t00 = time()
            p = sess.run(prediction, feed_dict=feed_dict)
            t11 = time()
            
            print '\t', t11 - t00 / 100.

            dset_p[i: i + 100] = p

        t1 = time()
        time_per_event = (t1 - t0)/ NUM_TRAIN_EVENTS
        print time_per_event
    ##########################################################################
