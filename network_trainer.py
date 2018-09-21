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

# import neural network model and debug mode
model, debug = import_model(only_model=False)

num_epochs = 1000 if not debug else 2
num_events = NUM_DEBUG_EVENTS if debug else NUM_TRAIN_EVENTS


# Loss & Training
# Compute cross entropy as loss function
with tf.name_scope(model.title):
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

def save_output(cost, acc=0, epoch=0):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file"""

    print "Epoch %s\tcost: %f\tacc: %f" % (epoch, cost, acc)

    with open(PATH + 'data/results/%s/epoch_cost_acc.txt' % (model.title), 'a') as f:
        f.write(str(epoch) + ',' + str(cost) + ',' + str(acc) + '\n')


def train_model(sess):
    """ trains model,
        input sess, a tensorflow session."""
    print 'Start training'
    batch_size = 20 
    for epoch in range(num_epochs):
        print "epoch", epoch 

        #######################################################################
        for batch, (events, labels) in enumerate(model.batches(batch_size=batch_size, debug=debug)):
            # Train
            feed_dict = {model.x: events, model.y: labels} 
            sess.run([optimizer], feed_dict=feed_dict)

            if batch % 100 == 0:
                acc, c = sess.run([accuracy, cost], feed_dict=feed_dict)
                save_output(c, acc, epoch)
                # Save weights every x events
                save_path = saver.save(sess, PATH + "weights/%s.ckpt" % model.title)
                print '\t save at', batch
        
        save_path = saver.save(sess, PATH + "weights/%s.ckpt" % model.title)
        ########################################################################


def main():
    # Session
    print 'Start session'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, PATH + "weights/%s.ckpt" % model.title)
        except:
            print 'Initalize variables'
            sess.run(tf.global_variables_initializer())
        train_model(sess)
 

if __name__ == "__main__":
    a = main()
