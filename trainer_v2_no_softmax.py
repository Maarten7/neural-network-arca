###################################
# Maarten Post
###################################
""" This neural network should take the detector slices and with shared weights
map them to an array or matrix and than this one goes to the output"""
import tensorflow as tf
import numpy as np
import datetime
import h5py
import sys
import importlib
from time import time
from helper_functions import * 

model = sys.argv[1].replace('/', '.')[:-3]
model = importlib.import_module(model)
title = model.title 

debug = eval(sys.argv[2])
num_epochs = 1000 if not debug else 2
num_events = NUM_DEBUG_EVENTS if debug else NUM_TRAIN_EVENTS

f = h5py.File(PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title, 'r')
#################################################################

# Loss & Training
# Compute cross entropy as loss function
with tf.device('/device:GPU:1'):
    output = model.cnn(model.x)
    prediction = tf.nn.softmax(output)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=model.y))
    # Train network with AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    # Compute the accuracy
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(model.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    saver = tf.train.Saver()

def save_output(acc, cost, test=False):
    
    mode = 'test' if test else 'train'
    if test:
        ne = NUM_TEST_EVENTS
    else:
        ne = num_events

    acc = acc / float(ne)
    cost_per_event = cost / float(ne)

    print '\t%s\tacc: %f\tcost: %f' % (mode, acc, cost_per_event) 
    with open(PATH + 'data/results/%s/acc_%s.txt' % (title + '_v2_no_softmax', mode), 'a') as f:
        f.write(str(acc) + '\n')
    with open(PATH + 'data/results/%s/cost_%s.txt' % (title + '_v2_no_softmax', mode), 'a') as f:
        f.write(str(cost_per_event) + '\n')


def test_model(sess):
    print 'Start testing'
    acc = 0  
    epoch_loss = 0

    for root_file, _ in root_files(train=False, test=True):
        events, labels = f[root_file].value, f[root_file + 'labels'].value
        
        feed_dict = {model.x: events, model.y: labels}
        a, c = sess.run([accuracy, cost], feed_dict=feed_dict)
        epoch_loss += c * len(labels)
        acc += a * len(labels) 
   
    save_output(acc, epoch_loss, test=True)
    return acc


def train_model(sess, test=True):
    print 'Start training'
    batch_size = 500
    for epoch in range(num_epochs):
        print "epoch", epoch 
        acc = 0 
        epoch_loss = 0

        #######################################################################
        for i, (root_file, _) in enumerate(root_files(debug=debug)):
            events, labels = f[root_file], f[root_file + 'labels']

#            for j in range(0, len(labels), batch_size):
#                feed_dict = {model.x: events[j: j + batch_size], model.y: labels[j: j + batch_size]} 
            feed_dict = {model.x: events.value, model.y: labels.value} 
            _, a, c = sess.run([optimizer, accuracy, cost], feed_dict=feed_dict)
            batch_size = len(labels)
            epoch_loss += c * batch_size
            acc += a * batch_size 
       
        save_output(acc, epoch_loss)
        if test and epoch % 10 == 0: test_model(sess)
        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % title + '_v2_no_softmax')
        ########################################################################
    return epoch_loss


def main():
    # Tensorflow Session
    print 'Start session'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        try:
            # restores trained weight so training can stop but also continue
            saver.restore(sess, PATH + "weights/%s.ckpt" % title + '_v2_no_softmax')
        except:
            print 'Initalize variables'
            sess.run(tf.global_variables_initializer())
        train_model(sess, test=True)
 


if __name__ == "__main__":
    t_start = time()
    a = main()
    t_end = time()
    print 'runtime', datetime.timedelta(seconds=t_end - t_start)
        
# dropout
# learning rate
# num conv layers
# num fc layers
# size of convolutions
# - weigts and biases
# activation 
# optimiers
# batch norm
