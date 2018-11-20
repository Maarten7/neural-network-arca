import tensorflow as tf
from tf_help import conv3d, maxpool3d, weight, bias
import batch_handle

title = 'multi_gpu'
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3
num_mini_timeslices = 200

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices
    mini_timeslices = tf.unstack(x, num_mini_timeslices, 1)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(3)])
    
    outputs, states = tf.contrib.rnn.static_rnn(stacked_lstm, mini_timeslices, dtype=tf.float32)
    output = tf.reshape(outputs[-1], [-1, 13 * 13 * 18 * 10 ])

    W = weight([13 * 13 * 18 * 10, NUM_CLASSES])
    b = bias(NUM_CLASSES)
    output = tf.matmul(output, W) + b

    return output 
    

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
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

        events, labels = batch_handle.toy_batches(batch_size=20).next()

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([events, labels], capacity=2 * 2)

        tower_grads = [] 
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(2):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        events_batch, labels_batch = batch_queue.dequeue()

                        loss = tower_loss(scope, events_batch, label_batch)

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
