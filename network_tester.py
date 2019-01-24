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

test_file = 'all_400ns_with_ATM_test'
# Tensorboard and saving variables
saver = tf.train.Saver()

# Session
def writer(out_file):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, PATH + "weights/%s.ckpt" % model.title)

        ##########################################################################
        print "Testing"
        hfile = h5py.File(PATH + 'results/%s/%s.hdf5' % (model.title, out_file), 'w')

        num_events = NUM_TEST_EVENTS
        dset_pred = hfile.create_dataset('all_test_predictions', shape=(num_events, model.NUM_CLASSES), dtype='float')

        i = 0
        batch_size = 30 
        for events, labels in batches(test_file, batch_size):
            print i

            feed_dict = {model.x: events, model.y: labels, model.keep_prob: 1.}
            p = sess.run(model.prediction, feed_dict=feed_dict)

            dset_pred[i: i + batch_size] = p

            i += batch_size

        hfile.close()

writer('all_400ns_with_ATM_test_result')
