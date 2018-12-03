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
from models.batch_handle import *

# import neural network model and debug mode
model, debug = import_model(only_model=False)

num_epochs = 1000 if not debug else 2
begin_epoch = 0

saver = tf.train.Saver()

def save_output(cost, acc=0, epoch=0, batch=0):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file"""

    print "Epoch %s\tcost: %f\tacc: %f\tbatch: %i" % (epoch, cost, acc, batch)

    with open(PATH + 'data/results/%s/epoch_cost_acc.txt' % (model.title), 'a') as f:
        f.write(str(epoch) + ',' + str(cost) + ',' + str(acc) + ',' + str(batch) + '\n')


def train_model(sess):
    """ trains model,
        input sess, a tensorflow session."""
    print 'Start training'
    batch_size = 15 
    for epoch in range(begin_epoch, num_epochs):

        #######################################################################
        np.random.seed()
        for batch, (events, labels) in enumerate(batches(batch_size=batch_size, debug=debug)):

            # Train
            feed_dict = {model.x: events, model.y: labels, model.keep_prob: .8, model.learning_rate: 0.003 * .93 ** epoch} 
            pred, _ = sess.run([model.prediction, model.train_op], feed_dict=feed_dict)

            if batch % 100 == 0:
                save_path = saver.save(sess, PATH + "weights/%s.ckpt" % model.title)
            if False:

                t_cost, t_acc = 0, 0
                for val_events, val_labels in get_validation_set():

                    
                    feed_dict = {model.x: val_events, model.y: val_labels, model.keep_prob: 1}
                    acc, cost = sess.run([model.accuracy, model.cost], feed_dict=feed_dict)

                    t_cost += cost
                    t_acc += acc
                
                # Save weights every x events
                save_output(t_cost / 40, t_acc / 40, epoch, batch)
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
    main()
