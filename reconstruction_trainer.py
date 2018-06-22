###################################
# Maarten Post
# trainer.py
###################################
import tensorflow as tf
import numpy as np
import datetime
import h5py
import sys
from time import time
from helper_functions import * 

model = import_model()
title = model.title
batches = model.batches

debug = eval(sys.argv[2])
num_epochs = 1000 if not debug else 2
num_events = NUM_DEBUG_EVENTS if debug else NUM_GOOD_TRAIN_EVENTS_3


# Loss & Training
# Compute cross entropy as loss function
with tf.name_scope(title):
    with tf.name_scope("Model"):
        output = model.cnn(model.x)
    with tf.name_scope("Xentropy"):
        cost = tf.reduce_sum(tf.square(output - model.y))
    # Train network with AdamOptimizer
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
    
    saver = tf.train.Saver()

def save_output(cost, test=False):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file
        input test, boolean default False, if true the acc and cost are of the 
            test set"""
    mode = 'test' if test else 'train'
    if test:
        ne = NUM_GOOD_TEST_EVENTS_3
    else:
        ne = num_events 

    cost_per_event = cost / float(ne)
    print "\t%s\tcost: %f" % (mode, cost_per_event)

    with open(PATH + 'data/results/%s/cost_%s.txt' % (title, mode), 'a') as f:
        f.write(str(cost_per_event) + '\n')

def test_model(sess):
    """ test model on test data set
        input sess: a tensor flow session"""
    epoch_loss = 0
    batch_size = 100 

    # loop over all data in batches
    for events, labels in test_batches(batch_size=batch_size):
        # Train
        feed_dict = {model.x: events, model.y: labels} 
        c = sess.run([cost], feed_dict=feed_dict)

        # Calculate loss and accuracy
        epoch_loss += c * batch_size

    save_output(epoch_loss, test=True)

def train_model(sess, test=True):
    """ trains model,
        input sess, a tensorflow session.
        input test, boolean, default True, if True the accuracy and cost
                    of test set are calculated"""
    print 'Start training'
    batch_size = 60 
    for epoch in range(num_epochs):
        print "epoch", epoch 
        epoch_loss = 0
        #######################################################################
        for events, labels in batches(batch_size=batch_size):
            # Train
            feed_dict = {model.x: events, model.y: labels} 
            _, c = sess.run([optimizer, cost], feed_dict=feed_dict)

            # Calculate loss and accuracy
            epoch_loss += c * batch_size 

        # Save accuracy and loss/cost
        save_output(epoch_loss)

        # test network and save weights
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
        train_model(sess, test=False)
 

if __name__ == "__main__":
    t_start = time()
    a = main()
    t_end = time()
    print 'runtime', str(datetime.timedelta(seconds=t_end - t_start)) 
