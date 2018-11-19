import tensorflow as tf

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    # size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, 
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding='SAME')


def weight(shape):
    w = tf.Variable(tf.contrib.layers.variance_scaling_initializer()(shape=shape), name="Weights")
    return w


def bias(shape):
    b = tf.Variable(tf.contrib.layers.variance_scaling_initializer()(shape=[shape]), name="Bias")
    return b

    
def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
