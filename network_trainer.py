###################################
# Maarten Post
# trainer.py
###################################
import tensorflow as tf
import numpy as np
import h5py

from helper_functions import * 
from data.batch_handle import *

# import neural network model and debug mode
model, debug = import_model(only_model=False)

num_epochs = 1000 if not debug else 2
begin_epoch = 2
print 'begin epoch', begin_epoch

saver = tf.train.Saver()

train_file = '400ns_ATM_BIG'
test_file = 'all_400ns_with_ATM_test'
validation_file = 'all_400ns_with_ATM_validation'

def train_model(sess):
    """ trains model,
        input sess, a tensorflow session."""
    print 'Start training'
    batch_size = 15 
    batch_size_val = 25 
    for epoch in range(begin_epoch, num_epochs):

        #######################################################################
        for batch, (events, labels) in enumerate(batches(train_file, batch_size)):

            # Train
            feed_dict = {model.x: events, model.y: labels, model.keep_prob: .8, model.learning_rate: 0.003 * .93 ** epoch} 
            pred, _ = sess.run([model.prediction, model.train_op], feed_dict=feed_dict)

            if batch % 5000 == 0:
                
                #### Validation
                t_cost, t_acc = 0, 0
                for val_events, val_labels in batches(validation_file, batch_size_val):
                    
                    feed_dict = {model.x: val_events, model.y: val_labels, model.keep_prob: 1}
                    acc, cost = sess.run([model.accuracy, model.cost], feed_dict=feed_dict)

                    t_cost += cost
                    t_acc += acc

                # Save weights every x events
                z = NUM_VAL_EVENTS / batch_size_val
                save_output(model.title, t_cost / z , t_acc / z, epoch, batch)

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
