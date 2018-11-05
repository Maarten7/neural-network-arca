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
