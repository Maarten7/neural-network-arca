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
from toy_model import *

title = 'temporal'
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3
num_mini_timeslices = 200

x = tf.placeholder(tf.float32, [None, num_mini_timeslices, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

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

def cnn(mini_timeslice):
    """ input: event tensor numpy shape 1, 13, 13, 18, 3"""
    conv1 = tf.nn.relu(
        conv3d(mini_timeslice, weights["l1"]) + biases["l1"])

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

    return fc

def km3nnet(x):
    """ input: event tensor numpy shape num_minitimeslices, 18, 18, 13, 3
        output: label prediction shape 3 (one hot encoded)"""
    # loop over mini time slices
    mini_timeslices = tf.unstack(x, num_mini_timeslices, 1)
    out_time_bin = []
    for ts in mini_timeslices:
        out_time_bin.append(cnn(ts))
    c = tf.concat(out_time_bin, 1)
    c = tf.reshape(c, [-1, num_mini_timeslices, nodes["l4"]])
    c = tf.unstack(c, num_mini_timeslices, 1)

    lstm_layer = tf.contrib.rnn.BasicLSTMCell(nodes["l5"], forget_bias=1.)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, c, dtype=tf.float32)

    output = tf.matmul(outputs[-1], weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    #conv_lstm_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=3, input_shape=[13, 13, 18, 3], output_channels=10, kernel_shape=[2, 2, 3])
#    outputs_1, _ = tf.contrib.rnn.static_rnn(conv_lstm_1, mini_timeslices, dtype='float32')
#
#    max_pool = maxpool3d(outputs_1[-1])
#
#    flatten = tf.reshape(max_pool, [-1, 7 * 7 * 9 * 10])
#
#    W = weight([7 * 7 * 9 * 10, NUM_CLASSES])
#    b = bias(NUM_CLASSES)
#    output = tf.matmul(flatten, W) + b

    return output 

output = km3nnet(x)
prediction = tf.nn.softmax(output)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.7)
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
        indices = np.random.permutation(NUM_TRAIN_EVENTS)
        num_events = NUM_TRAIN_EVENTS

    for k in range(0, num_events, batch_size):
        if k + batch_size > num_events:
            batch_size = num_events - k

        batch = sorted(list(indices[k: k + batch_size]))
        
        tots = f['all_tots'][batch]
        bins = f['all_bins'][batch]
        labels = f['all_labels'][batch]

        num_hits = [len(tot) for tot in tots]
        extra_bin = np.concatenate([np.full(shape=j, fill_value=i) for i, j in zip(range(batch_size), num_hits)])
        tots = np.concatenate(tots)

        bins = [np.concatenate(bins[:,i]) for i in range(5)]
        bins.insert(0, extra_bin)
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        events[bins] = tots

        yield events, labels

def get_validation_set(validation_set_size=500):
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    indices = range(NUM_TRAIN_EVENTS, NUM_EVENTS)
    np.random.seed(0)
    indices = np.random.choice(indices, validation_set_size, replace=False)
    np.random.seed()
    batch = indices
    events = np.zeros((validation_set_size, num_mini_timeslices, 13, 13, 18, 3))
    labels = np.zeros((validation_set_size, NUM_CLASSES))
    for i, j in enumerate(batch):
        # get event bins and tots
        labels[i] = f['all_labels'][j]
        tots, bins = f['all_tots'][j], f['all_bins'][j]

        bins = tuple(bins)
        events[i][bins] = tots
    return events, labels

def toy_batches(batch_size, debug=False):
    while True:
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        labels = np.zeros((batch_size, 3))
        for i in range(batch_size):
            event, label = make_toy()
            events[i] = event
            labels[i] = label
        yield events, labels 

def get_toy_validation_set(size=500):
    np.random.seed(4)
    events, labels = toy_batches(size).next()
    np.random.seed()
    return events, labels

    
def see_slices(j, movie=True):
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    tots =  f['all_tots'][j]
    bins =  f['all_bins'][j]
    energy = f['all_energies'][j]
    evt_type = f['all_types'][j]
    num_hits = f['all_num_hits'][j]

    event = np.zeros((num_mini_timeslices, 13, 13, 18, 3))
    event[tuple(bins)] = tots

    if movie:
        animate_event(event) 
    else:
        event = np.sqrt(np.sum(np.square(event), axis=4))
        for minislice in event:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([0,13])
            ax.set_ylim([0,13])
            ax.set_zlim([0,18])
            ax.set_xlabel('x index')
            ax.set_ylabel('y index')
            ax.set_zlabel('z index')
            plt.title('TTOT on DOM E=%.1f  nh=%i  type=%s' % (energy, num_hits, EVT_TYPES[evt_type]))
            x, y, z = minislice.nonzero() 
            k = minislice[minislice.nonzero()]
            sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
            fig.colorbar(sc)
            plt.show()
    return 0

def animate_event(event_full):
    """Shows 3D plot of evt"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    event_full = np.sqrt(np.sum(np.square(event_full), axis=4))

    ims = []
    for i, event in enumerate(event_full):
        x, y, z = event.nonzero()
        k = event[event.nonzero()]
        print k
        sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
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
