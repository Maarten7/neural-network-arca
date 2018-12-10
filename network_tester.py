###################################
# Maarten Post
# network tester
###################################
""" This program takes a trained NN and runs it over the test set.
    Then it writes all the output to file. The output is the softmax threevector
    for each evetn"""
import tensorflow as tf
import h5py

from helper_functions import *
from data.batch_handle import batches

model = import_model()

# Tensorboard and saving variables
saver = tf.train.Saver()

# Session
def writer():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, PATH + "weights/%s.ckpt" % model.title)

        num_events = 81908 
        ##########################################################################
        print "Testing"
        with h5py.File(PATH + 'data/results/%s/20000ns_test_result_%s_no_threshold.hdf5' % (model.title, model.title), 'w') as hfile:
            dset_pred = hfile.create_dataset('all_test_predictions', shape=(num_events, model.NUM_CLASSES), dtype='float')
            i = 0
            batch_size = 30 
            for events, labels in batches(batch_size, test=True):

                feed_dict = {model.x: events, model.y: labels, model.keep_prob: 1.}
                p = sess.run(model.prediction, feed_dict=feed_dict)

                dset_pred[i: i + batch_size] = p

                i += batch_size

writer()
