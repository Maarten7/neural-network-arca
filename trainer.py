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

# import neural network model
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
        output = model.km3nnet(model.x)
        prediction = tf.nn.softmax(output)
    with tf.name_scope("Xentropy"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=model.y))
    # Train network with AdamOptimizer
    with tf.name_scope("Train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
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
        ne = NUM_GOOD_TEST_EVENTS_3
    else:
        ne = num_events 

    cost_per_event = cost / float(ne)
    acc = acc / float(ne)
    print "\t%s\tacc: %f\tcost: %f" % (mode, acc, cost_per_event)

    with open(PATH + 'data/results/%s/acc_%s.txt' % (title, mode), 'a') as f:
        f.write(str(acc) + '\n')
    with open(PATH + 'data/results/%s/cost_%s.txt' % (title, mode), 'a') as f:
        f.write(str(cost_per_event) + '\n')

def train_model(sess):
    """ trains model,
        input sess, a tensorflow session.
        input test, boolean, default True, if True the accuracy and cost
                    of test set are calculated"""
    print 'Start training'
    batch_size = 20 
    for epoch in range(num_epochs):
        print "epoch", epoch 

        #######################################################################
        batch = 0
        for events, labels in batches(batch_size=batch_size, debug=debug):
            batch += 1
            # Train
            feed_dict = {model.x: events, model.y: labels} 
            sess.run([optimizer], feed_dict=feed_dict)

            if batch % 6000 == 0:
                # Save weights every 2000 events
                save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title)
                print '\t save at', batch
        
        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title)
        ########################################################################


def main():
    # Session
    print 'Start session'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, PATH + "weights/%s.ckpt" % title)
        except:
            print 'Initalize variables'
            sess.run(tf.global_variables_initializer())
        train_model(sess)
 

if __name__ == "__main__":
    a = main()
