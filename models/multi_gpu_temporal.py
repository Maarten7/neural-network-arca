import tensorflow as tf
import numpy as np
from tf_help import conv3d, maxpool3d, weight, bias
import batch_handle
from helper_functions import NUM_CLASSES

title = 'multi_gpu'
num_mini_timeslices = 200
from toy_model import make_toy
from tf_help import *
NUM_CLASSES = 3

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices
    nodes =   {"l1": 25,
               "l2": 25,
               "l3": 80,
               "l4": 40,
               "l5": 20} 
               
    weights = {"l1": weight([4, 4, 4, 3, nodes["l1"]]),
               "l2": weight([3, 3, 3, nodes["l1"], nodes["l2"]]),
               "l3": weight([11025, nodes["l3"]]),
               "l4": weight([nodes["l3"], nodes["l4"]])}

    biases =  {"l1": bias(nodes["l1"]),
               "l2": bias(nodes["l2"]),
               "l3": bias(nodes["l3"]),
               "l4": bias(nodes["l4"])}

    conv1 = tf.nn.relu(
        conv3d(x, weights["l1"]) + biases["l1"])

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv2 = maxpool3d(conv2)

    elements = np.prod(conv2._shape_as_list()[1:])

    fc = tf.reshape(conv2, [-1, elements])
    
    fc = tf.nn.relu(
        tf.matmul(fc, weights["l3"]) + biases["l3"])

    fc = tf.nn.relu(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    fc = tf.reshape(fc, [50, 1, 40])
    c = tf.unstack(fc, num_mini_timeslices, 0)
    
    lstm_layer = tf.contrib.rnn.BasicLSTMCell(nodes["l5"], forget_bias=1.)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, c, dtype=tf.float32)

   output = tf.matmul(outputs[-1], weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    
    return tf.reshape(output, [3])


class toy_gen:
    def __init__(self, title):
        self.title = title

    def __call__(self):
        while True:
            e, l = make_toy(50)
            yield e, l  
            

data_set = tf.data.Dataset.from_generator(
    generator=toy_gen, 
    output_types=(tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([50, 13, 13, 18, 3]), tf.TensorShape([3])))


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, events, labels):
    logits = km3nnet(events) 
    _ = loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)

            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grads_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

    
def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        lr = 0.03
        opt = tf.train.GradientDescentOptimizer(lr)

        events, labels = data_set.make_one_shot_iterator().get_next()

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([events, labels], capacity=2 * 2)

        tower_grads = [] 
        with tf.variable_scope(tf.get_variable_scope()):
            for i in [1, 0]:
                with tf.device('/GPU:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        events_batch, labels_batch = batch_queue.dequeue()

                        loss = tower_loss(scope, events_batch, labels_batch)

                        tf.get_variable_scope().reuse_variables()

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        train_op = apply_gradient_op

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        _, loss_value = sess.run([train_op, loss])

train()
