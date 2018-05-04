import numpy as np
import tensorflow as tf
import random

def down():
	event = np.zeros((10, 10, 10))
	i = random.randint(0,5)
	j = random.randint(0,9)
	k = 0
	event[k][i][j] = 1
	
	while i < 10:
		k += 1
		try:
			i += 1
			s = random.randint(-1,1)
			j += s
			event[k][i][j] = .5
		except IndexError:
			break
	return event


def up():
	event = np.zeros((10, 10, 10))
	i = random.randint(5,9)
	j = random.randint(0,9)
	k = 0
	event[k][i][j] = 1
	
	while i > 0:
		k += 1
		try:
			i -= 1
			s = random.randint(-1,1)
			j += s
			event[k][i][j] = .5
		except IndexError:
			break
	return event

def weights(shape):
    w = tf.Variable(tf.random_normal(shape=shape), name="Weights")
    return w

def bias(shape):
    b = tf.Variable(tf.random_normal(shape=[shape]), name="Bias")
    return b

x = tf.placeholder(tf.float32, [None, 10, 10, 10, 1], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, 2], name="Y_placeholder")
print x.shape


m = []
for i in range(10):
    conv1 = tf.nn.conv2d(input=x[:,i,:,:,:], filter=weights([4,4, 1, 10]), strides=[1,1,1,1], padding='VALID')
    conv1 = conv1 + bias(10)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,1,1,1], strides=[1,1,1,1], padding='VALID') 

    elements = np.prod(conv1._shape_as_list()[1:])
    fc = tf.reshape(conv1, [-1, elements])

    fc = tf.matmul(fc, weights([elements, 10]))
    fc = fc + bias(10)
    fc = tf.nn.relu(fc)

    fc = tf.matmul(fc, weights([10, 16]))
    fc = fc + bias(16)
    fc = tf.nn.relu(fc)
    print fc.shape
    m.append(fc)
c = tf.concat(m, 1)
print c

time_steps = 10
num_units = 64
n_input = 10 #*10
n_classes = 2

lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, c, dtype=tf.float32)
prediction = tf.matmul(outputs[-1], weights([num_units, num_classes])) + bias(num_classes)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
opt = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initalizer()
with tf.Session() as sess:
	sess.run(init)
		
	for epoch in range(100):
		
		feed_dict = {x: [up() for _ in range(100)], y: [[1, 0] for _ in range(100)]}
		sess.run(opt, feed_dict=feed_dict)
		feed_dict = {x: [down() for _ in range(100)], y: [[0, 1] for _ in range(100)]}
		sess.run(opt, feed_dict=feed_dict)
	
	feed_dict = {x: [up() for _ in range(100)], y: [[1, 0] for _ in range(100)]}
	sess.run(accuracy, feed_dict=feed_dict)
	feed_dict = {x: [down() for _ in range(100)], y: [[0, 1] for _ in range(100)]}
	sess.run(accuracy, feed_dict=feed_dict)
	



