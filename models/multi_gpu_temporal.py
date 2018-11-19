import os.path
import tensorflow as tf

def inference(events):
    pass

def loss(logits, labels):
    pass

def tower_loss(scope, events, labels):
    pass

    logits = inference(events) 
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

        events, labels =  

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([events, labels], capacity=2 * 2)

        tower_grads = 
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
