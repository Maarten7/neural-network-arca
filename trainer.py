###################################
# Maarten Post
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

debug = eval(sys.argv[2])
num_epochs = 1000 if not debug else 1000 
num_debug_events = 500 * 3 * 2 * 4
num_events = num_debug_events if debug else NUM_TRAIN_EVENTS

f = h5py.File(PATH + 'data/hdf5_files/events_and_labels2_%s.hdf5' % title, 'r')

# Loss & Training
# Compute cross entropy as loss function
with tf.name_scope(title):
    with tf.name_scope("Model"):
        output = model.cnn(model.x)
        prediction = tf.nn.softmax(output)
    with tf.name_scope("Xentropy"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=model.y))
    # Train network with AdamOptimizer
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(cost)
    # Compute the accuracy
    with tf.name_scope("Test"):
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    saver = tf.train.Saver()

def save_output(acc, cost, test=False):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file
        input test, boolean default False, if true the acc and cost are of the 
            test set"""
    mode = 'test' if test else 'train'
    if test:
        ne = NUM_TEST_EVENTS
    else:
        ne = num_events 

    cost_per_event = cost / float(ne)
    acc = acc / float(ne)
    print "%s acc: %f, cost: %f" % (mode, acc, cost_per_event)

    with open(PATH + 'data/results/%s/acc_%s.txt' % (title, mode), 'a') as f:
        f.write(str(acc) + '\n')
    with open(PATH + 'data/results/%s/cost_%s.txt' % (title, mode), 'a') as f:
        f.write(str(cost_per_event) + '\n')

def test_model(sess):
    """ test model on test data set
        input sess: a tensor flow session"""
    acc = 0 
    epoch_loss = 0

    for root_file, _ in root_files(train=False, test=True):
        print root_file
        tots, bins, labels = f[root_file].value, f[root_file + 'labels'].value
        for j in range(len(labels)):
 
            t = tots[j]
            b = tuple(bins[j])

            events = np.empty((1, 50, 13, 13, 18, 1))
            event[b] = t

            feed_dict = {model.x: event, model.y: labels[j]} 
 
            p, a, c = sess.run([prediction, accuracy, cost], feed_dict=feed_dict)
            epoch_loss += c
            acc += a * len(labels)

    save_output(acc, epoch_loss, test=True)
    return acc

def train_model(sess, test=True):
    """ trains model,
        input sess, a tensorflow session.
        input test, boolean, default True, if True the accuracy and cost
                    of test set are calculated"""
    print 'Start training'
    for epoch in range(num_epochs):
        print "epoch", epoch 
        acc = 0
        epoch_loss = 0

        #######################################################################
        for i, (root_file, _) in enumerate(root_files(debug=debug)):

            # Create batches
            tots, bins, labels = f[root_file + 'tots'], f[root_file + 'bins'], f[root_file + 'labels']
            for j in range(0, len(labels), 100):
                events = np.empty((100, 50, 13, 13, 18, 3))
                for jj in range(j, j + 100):
                    b = tuple(bins[jj])
                    events[jj % 100][b] = tots[jj][0]

                # Train
                feed_dict = {model.x: events, model.y: labels[j: j + 100]} 
                _, c, p, a = sess.run([optimizer, cost, prediction, accuracy], feed_dict=feed_dict)

                # Calculate loss and accuracy
                epoch_loss += c
                acc += a * 100 
      
        # Save accuracy and loss/cost
        save_output(acc, epoch_loss)

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
