from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import h5py

from helper_functions import NUM_TRAIN_EVENTS, NUM_TEST_EVENTS, NUM_DEBUG_EVENTS, PATH, NUM_EVENTS
from detector_handle import pmt_to_dom_index, pmt_direction, hit_to_pmt, hit_time_to_index
from tf_help import conv3d, maxpool3d, weight, bias

title = 'temporal'
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3
num_mini_timeslices = 200

x = tf.placeholder(tf.float32, [None, num_mini_timeslices, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")
keep_prob = tf.placeholder(tf.float32)

nodes =   {"l1": 25,
           "l2": 35,
           "l3": 80,
           "l4": 40,
           "l5": 20} 
           
weights = {"l1": weight([4, 4, 4, 3, nodes["l1"]]),
           "l2": weight([3, 3, 3, nodes["l1"], nodes["l2"]]),
           "l3": weight([15435, nodes["l3"]]),
           "l4": weight([nodes["l3"], nodes["l4"]])}

biases =  {"l1": bias(nodes["l1"]),
           "l2": bias(nodes["l2"]),
           "l3": bias(nodes["l3"]),
           "l4": bias(nodes["l4"])}

def cnn(input_slice):
    """ input: event tensor numpy shape 1, 13, 13, 18, 3"""
    conv1 = tf.nn.relu(
        conv3d(input_slice, weights["l1"]) + biases["l1"])

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv2 = maxpool3d(conv2)

    elements = np.prod(conv2._shape_as_list()[1:])

    fc = tf.reshape(conv2, [-1, elements])
    
    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l3"]) + biases["l3"])

    fc = tf.nn.dropout(fc, keep_prob)

    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    return fc

def km3nnet(x):
    """ input: event tensor numpy shape 400, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    out_time_bin = [] 
    # loop over 400 time slices
    for i in range(num_mini_timeslices):
        input_slice = x[:,i,:,:,:,:] 

        conv1 = tf.nn.relu(
            conv3d(input_slice, weights["l1"]) + biases["l1"])

        conv2 = tf.nn.relu(
            conv3d(conv1, weights["l2"]) + biases["l2"])

        conv2 = maxpool3d(conv2)

        elements = np.prod(conv2._shape_as_list()[1:])

        fc = tf.reshape(conv2, [-1, elements])
        
        fc = tf.nn.relu(
            tf.matmul(fc, weights["l3"]) + biases["l3"])

        fc = tf.nn.dropout(fc, keep_prob)

        fc = tf.nn.relu(
            tf.matmul(fc, weights["l4"]) + biases["l4"])
        
        out_time_bin.append(fc)

    c = tf.concat(out_time_bin, 1)
    c = tf.reshape(c, [-1, num_mini_timeslices, 40])
    c = tf.unstack(c, num_mini_timeslices, 1)

    lstm_layer = tf.contrib.rnn.BasicLSTMCell(nodes["l5"], forget_bias=1.)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, c, dtype=tf.float32)

    output = tf.matmul(outputs[-1], weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    return output 

output = km3nnet(x)
prediction = tf.nn.softmax(output)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
#gvs = optimizer.compute_gradients(cost)
#capped_gvs= [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_op = optimizer.apply_gradients(capped_gvs)
train_op = optimizer.minimize(cost)


correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


def make_event(hits, norm_factor=100, tot_mode=True, tbin_size=50):
    "Take aa_net hits and put them in cube numpy arrays"

    event = np.zeros((20000 / tbin_size, 13, 13, 18, 3))

    for hit in hits:

        tot       = hit.tot if tot_mode else 1

        pmt       = hit_to_pmt(hit)

        direction = pmt_direction(pmt)

        x, y, z   = pmt_to_dom_index(pmt)

        t         = hit_time_to_index(hit, tbin_size)

        event[t, x, y, z] += direction * tot / norm_factor 
            
    non = event.nonzero()
    return event[non], np.array(non)

def make_labels(code):
    """ Makes one hot labels form evt_type code"""
    if code < 4:
        return np.array([1, 0, 0])
    elif code < 6:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])
        
def batches(batch_size, test=False, debug=False):
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    if debug:
        indices = np.random.choice(NUM_TRAIN_EVENTS, NUM_DEBUG_EVENTS, replace=False)
        num_events = NUM_DEBUG_EVENTS 
    elif test:
        indices = range(NUM_TRAIN_EVENTS, NUM_EVENTS)
        num_events = NUM_TEST_EVENTS
    else:
        indices = np.random.choice(NUM_TRAIN_EVENTS, NUM_TRAIN_EVENTS, replace=False)
        num_events = NUM_TRAIN_EVENTS

    for k in range(0, num_events, batch_size):
        if k + batch_size > num_events:
            batch_size = k + batch_size - num_events
        batch = indices[k: k + batch_size]

        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            # get event bins and tots
            labels[i] = f['all_labels'][j]
            tots, bins = f['all_tots'][j], f['all_bins'][j]

            bins = tuple(bins)
            events[i][bins] = tots

        yield events, labels

def animate_event(event_full):
    """Shows 3D plot of evt"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    event_full = np.sqrt(np.sum(np.square(event_full), axis=4))

    ims = []
    for i, event in enumerate(event_full):
        x, y, z = event.nonzero()
        k = event[event.nonzero()]
        sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
        ax.text2D(0, 0, i)
        ims.append([sc])
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_zlim([0,18])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_zlabel('z index')
    plt.title('TTOT on DOM')
    fig.colorbar(sc)
    ani = animation.ArtistAnimation(fig, ims)
    #writer = animation.writers['ffmpeg']
    plt.show()

if __name__ == "__main__":
    pass
