from ROOT import *
import aa
import numpy as np
import tensorflow as tf
from helper_functions import *

title = 'three_classes_sum_tot'
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def maxpool3d(x):
    # size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, 
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding='SAME')

def weight(shape):
    w = tf.Variable(tf.random_normal(shape=shape), name="Weights")
    return w

def bias(shape):
    b = tf.Variable(tf.random_normal(shape=[shape]), name="Bias")
    return b

x = tf.placeholder(tf.float32, [None, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="Y_placeholder")

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

    conv3 = tf.nn.relu(
        conv3d(conv2, weights["l3"] + biases["l3"]))

    elements = np.prod(conv3._shape_as_list()[1:])
    fc = tf.reshape(conv3, [-1, elements])
    
    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l4"]) + biases["l4"])

    fc = tf.nn.sigmoid(
        tf.matmul(fc, weights["l5"]) + biases["l5"])
    return fc

def km3nnet(x):
    x = tf.reshape(x, shape=[-1, 13, 13, 18, 3])

    fc  = cnn(x) 

    output = tf.matmul(fc, weights([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    return output

class Data_handle(object):
    def __init__(self, tot_mode=True, norm=100):
        self.lines = self.lines()
        self.z_index = self.z_index()
        self.det = Det(PATH + 'data/km3net_115.det')
        self.NORM_FACTOR = float(norm) 
        self.TOT_MODE = tot_mode

    def lines(self):
        return np.array([
                [0,   0,   109, 110, 111, 0,   0,  0,   0,   0,   0,   0,   0 ],
                [0,   81,  82,  83,  84,  85,  86, 0,   0,   0,   0,   0,   0 ],
                [108, 80,  53,  54,  55,  56,  57, 87,  112, 0,   0,   0,   0 ],
                [107, 79,  52,  31,  32,  33,  34, 58,  88,  113, 0,   0,   0 ],
                [106, 78,  51,  30,  15,  16,  17, 35,  59,  89,  114, 0,   0 ],
                [0,   77,  50,  29,  14,  5,   6,  18,  36,  60,  90,  115, 0 ],
                [0,   76,  49,  28,  13,  4,   1,  7,   19,  37,  61,  91,  0 ],
                [0,   105, 75,  48,  27,  12,  3,  2,   10,  24,  44,  70,  0 ],
                [0,   0,   104, 74,  47,  26,  11, 9,   8,   22,  42,  68,  99],
                [0,   0,   0,   103, 73,  46,  25, 23,  21,  20,  40,  66,  97],
                [0,   0,   0,   0,   102, 72,  45, 43,  41,  39,  38,  64,  95],
                [0,   0,   0,   0,   0,   101, 71, 69,  67,  65,  63,  62,  93],
                [0,   0,   0,   0,   0,   0,   0,  100, 98,  96,  94,  92,  0 ]
                ])


    def z_index(self):
        return {712: 0, 676: 1, 640: 2, 604: 3, 568: 4, 532: 5, 496: 6,
              460: 7, 424: 8, 388: 9, 352: 10, 316: 11, 280: 12,
              244: 13, 208: 14, 172: 15, 136: 16, 100: 17}

    def line_to_index(self, line):
        """ Takes a line number and return it's
        position and it's slice. Where event are made like
        backslashes"""
        i, j = np.where(self.lines == line)
        return np.int(i), np.int(j)

    def make_event(self, hits):
        "Take aa_net hits and put them in cube numpy arrays"
        event = np.zeros((13, 13, 18, 3))
        for hit in hits:
            tot = hit.tot if self.TOT_MODE else 1
            pmt = self.det.get_pmt(hit.dom_id, hit.channel_id)
            direction = np.array([pmt.dir.x, pmt.dir.y, pmt.dir.z])
            dom = self.det.get_dom(pmt)
            line_id = dom.line_id
            # also valid
            # line_id = np.ceil(hit.dom_id / 18.)

            z = self.z_index[round(dom.pos.z)] 
            y, x = self.line_to_index(dom.line_id)

            event[x, y, z] += direction * tot / self.NORM_FACTOR 
        non = event.nonzero()
        return event[non], np.array(non)

    def make_labels(self, code):
        """ Makes one hot labels form evt_type code"""
        if code < 4:
            return np.array([1, 0, 0])
        elif code < 6:
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])


f = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')
def batches(batch_size, test=False, debug=False):
    if debug:
        indices = np.random.choice(NUM_GOOD_TRAIN_EVENTS_3, NUM_DEBUG_EVENTS, replace=False)
        num_events = NUM_DEBUG_EVENTS 
    elif test:
        indices = range(NUM_GOOD_TRAIN_EVENTS_3, NUM_GOOD_EVENTS_3)
        num_events = NUM_GOOD_TEST_EVENTS_3
    else:
        indices = np.random.choice(NUM_GOOD_TRAIN_EVENTS_3, NUM_GOOD_TRAIN_EVENTS_3, replace=False)
        num_events = NUM_GOOD_TRAIN_EVENTS_3

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
    det = Det(PATH + 'data/km3net_115.det')
    plt.plot(0,0)
    plt.axes().set_aspect('equal' )
    xs, ys = [], []
    for hit in hits:
        pmt = det.get_pmt(hit.dom_id, hit.channel_id)
        dom = det.get_dom(pmt)
        xs.append(dom.pos.x)
        ys.append(dom.pos.y)
    print xs, ys 
    plt.plot(xs, ys, '.')
    plt.show()

if __name__ == "__main__":
    pass
