###################################
# Maarten Post
###################################

import tensorflow as tf
import numpy as np
import h5py
import sys
import importlib
import matplotlib.pyplot as plt

from helper_functions import *

model = sys.argv[1] 
model = importlib.import_module(model)
x, y = model.x, model.y
title = model.title
cnn = model.cnn

# Train data
f = h5py.File(PATH + 'data/hdf5_files/bg_file_%s.hdf5' % title, 'r')
# Write file
z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'w')

# Tensorboard and saving variables
with tf.name_scope(title):
    with tf.name_scope("Model"):
        prediction = cnn(x)
saver = tf.train.Saver()

# Session
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config) as sess:
    saver.restore(sess, PATH + "weights/%s.ckpt" % title)

    ##########################################################################
    print "Testing with bg"

    pred = np.empty((0, NUM_CLASSES))
    labe = np.empty((0, NUM_CLASSES))
    even = np.empty((0, 1, 13, 13, 18))
    for root_file, _ in root_files(train=False, test=True):
        events, labels = f[root_file].value, f[root_file + 'labels'].value
        feed_dict = {x: events, y: labels}
        p = sess.run(prediction, feed_dict=feed_dict)

        pred = np.append(pred, p, axis=0)
        labe = np.append(labe, labels, axis=0)
        even = np.append(even, events, axis=0)

    z.create_dataset('predictions_bg', data=pred)
    z.create_dataset('labels_bg', data=labe)
    z.create_dataset('events_bg', data=even)
    ##########################################################################
