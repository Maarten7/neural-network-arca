import numpy as np
import tensorflow as tf
import random

def weights(shape):
    w = tf.Variable(tf.random_normal(shape=shape), name="Weights")
    return w


def bias(shape):
    b = tf.Variable(tf.random_normal(shape=[shape]), name="Bias")
    return b


x = tf.placeholder(tf.float32, [None, 10, 12, 14, 1], name="X")
y = tf.placeholder(tf.float32, [None, 2], name="Y")


m = []
wconv = weights([4, 4, 1, 5])
bconv = bias(5) 
wfc1 = weights([495, 10])
bfc1 = bias(10)
wfc2 = weights([10, 16])
bfc2 = bias(16)

for i in range(10):
    print "##############################################"
    print x.shape
    print x[:, i, :, :].shape
    conv1 = tf.nn.conv2d(input=x[:, i, :, :], filter=wconv, strides=[1, 1, 1, 1], padding='VALID') + bconv
    conv1 = tf.nn.relu(conv1)
    print conv1.shape
    conv1 = tf.nn.max_pool(
        conv1, ksize=[
            1, 1, 1, 1], strides=[
            1, 1, 1, 1], padding='VALID')
    print conv1.shape

    elements = np.prod(conv1._shape_as_list()[1:])
    fc = tf.reshape(conv1, [-1, elements])
    print fc.shape

    fc = tf.matmul(fc, wfc1) + bfc1 
    fc = tf.nn.relu(fc)
    print fc.shape

    fc = tf.matmul(fc, wfc2) + bfc2 
    fc = tf.nn.relu(fc)
    print fc.shape
    fc = tf.reshape(fc, shape=[-1, 1, 16, 1])
    print fc.shape
    m.append(fc)

c = tf.concat(m, 1)

print c

wconv2 = weights([4, 16, 1, 7])
bconv2 = bias(7)
conv2 = tf.nn.conv2d(input=c, filter=wconv2, strides=[1, 1, 1, 1], padding="VALID") + bconv2
conv2 = tf.nn.relu(conv2)
print conv2
conv2 = tf.nn.max_pool(conv2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
print conv2
elements = np.prod(conv2._shape_as_list()[1:])
fc2 = tf.reshape(conv2, [-1, elements]) 
print fc2
wfc3 = weights([elements, 16])
bfc3 = bias(16)
fc2 = tf.matmul(fc2, wfc3) + bfc3
fc2 = tf.nn.relu(fc2)
print fc2
wfc4 = weights([16, 2])
bfc4 = bias(2)
fc2 = tf.matmul(fc2, wfc4) + bfc4
print fc2
prediction = tf.nn.sigmoid(fc2)
print prediction

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '/tmp/updownmodel_cnn.ckpt')

    for epoch in range(1):
        if epoch % 10 == 0:
            print epoch
        feed_dict = {x: events, y: labels}
        sess.run(opt, feed_dict=feed_dict)
        cc, p = sess.run([c, prediction], feed_dict=feed_dict)

        
        if epoch % 100 == 0:
            saver.save(sess, '/tmp/updownmodel_cnn.ckpt')
            feed_dict = {x: tevents, y: tlabels}
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print acc
