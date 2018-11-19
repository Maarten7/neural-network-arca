from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import h5py

from helper_functions import NUM_TRAIN_EVENTS, NUM_TEST_EVENTS, NUM_DEBUG_EVENTS, PATH
from detector_handle import pmt_to_dom_index, pmt_direction, hit_to_pmt
from tf_help import conv3d, maxpool3d, weight, bias

title = 'three_classes_sum_tot'
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3


x = tf.placeholder(tf.float32, [None, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")

nodes =   {"l1": 25,
           "l2": 35,
           "l3": 80,
           "l4": 40,
           "l5": 20} 
           
weights = {"l1": weight([4, 4, 4, 3, nodes["l1"]]),
           "l2": weight([3, 3, 3, nodes["l1"], nodes["l2"]]),
           "l3": weight([3, 3, 3, nodes["l2"], nodes["l3"]]),
           "l4": weight([7 * 7 * 9 * nodes["l3"], nodes["l4"]]),
           "l5": weight([nodes["l4"], nodes["l5"]])}

biases =  {"l1": bias(nodes["l1"]),
           "l2": bias(nodes["l2"]),
           "l3": bias(nodes["l3"]),
           "l4": bias(nodes["l4"]),
           "l5": bias(nodes["l5"])}

def cnn(input_slice):
    """ input: event tensor numpy shape 1, 13, 13, 18, 3"""
    conv1 = tf.nn.relu(
        conv3d(input_slice, weights["l1"]) + biases["l1"])

    conv2 = tf.nn.relu(
        conv3d(conv1, weights["l2"]) + biases["l2"])

    conv2 = maxpool3d(conv2)

    conv3 = tf.nn.relu(
        conv3d(conv2, weights["l3"] + biases["l3"]))

    fc = tf.reshape(conv3, [-1, 7 * 7 * 9 * nodes["l3"]])
    
    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l5"]) + biases["l5"])
    return fc

def km3nnet(x):
    x = tf.reshape(x, shape=[-1, 13, 13, 18, 3])

    fc  = cnn(x) 

    output = tf.matmul(fc, weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    return output

output = km3nnet(x)


def make_event(hits, norm_factor=100, tot_mode=True):
    "Take aa_net hits and put them in cube numpy arrays"
    event = np.zeros((13, 13, 18, 3))
    for hit in hits:

        tot       = hit.tot if tot_mode else 1

        pmt       = hit_to_pmt(hit)

        direction = pmt_direction(pmt)

        x, y, z   = pmt_to_dom_index(pmt)

        event[x, y, z] += direction * tot / tot_mode 

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
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
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
        events = np.zeros((batch_size, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            # get event bins and tots
            labels[i] = f['all_labels'][j]
            tots, bins = f['all_tots'][j], f['all_bins'][j]
            events[i][tuple(bins)] = tots 
        yield events, labels

def show_event(event):
    """Shows 3D plot of evt"""
    assert np.shape(event) == (13, 13, 18)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = event.nonzero()
    k = event[event.nonzero()]
    #sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=plt.Normalize(0, 100))
    #sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'))#, norm=plt.Normalize(0, 100))
    sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_zlim([0,18])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_zlabel('z index')
    plt.title('TTOT on DOM')
    fig.colorbar(sc)
    plt.show()

def show_event2d(event):
    """Shows 3D plot of evt"""
    assert np.shape(event) == (13, 13, 18)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y, z = event.nonzero()
    ax.scatter(*np.ones((13,13)).nonzero())
    sc = ax.scatter(x, y)
    ax.set_xlim([-1,13])
    ax.set_ylim([-1,13])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    plt.axes().set_aspect('equal' )
    plt.show()

def show_event_pos(hits):
    plt.plot(0,0)
    plt.axes().set_aspect('equal' )
    xs, ys = [], []
    for hit in hits:
        pmt = detector_handle.det.get_pmt(hit.dom_id, hit.channel_id)
        dom = detector_handle.det.get_dom(pmt)
        xs.append(dom.pos.x)
        ys.append(dom.pos.y)
    print xs, ys 
    plt.plot(xs, ys, '.')
    plt.show()

if __name__ == "__main__":
    cnn(x) 
