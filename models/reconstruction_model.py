from ROOT import *
import aa
import numpy as np
import tensorflow as tf
from helper_functions import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import h5py
title = 'reconstruction'

EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 7 # energie, x, y, z, dx, dy, dz

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

def print_tensor(x):
    #print 'x\t\t', x.shape, np.prod(x._shape_as_list()[1:])
    pass

x = tf.placeholder(tf.float32, [None, 50, 13, 13, 18, 3], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, 3], name="Y_placeholder")

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

def cnn(x):
    print_tensor(x)
    out_time_bin = []
    time_bins = x._shape_as_list()[1]
    #time_bins = tf.shape(x)[1]
    for i in range(time_bins):
        input = x[:,i,:,:,:,:] 
        conv1 = tf.nn.relu(
            conv3d(input, weights["l1"]) + biases["l1"])

        conv2 = tf.nn.relu(
            conv3d(conv1, weights["l2"]) + biases["l2"])

        conv2 = maxpool3d(conv2)

        elements = np.prod(conv2._shape_as_list()[1:])

        fc = tf.reshape(conv2, [-1, elements])
        
        fc = tf.nn.sigmoid(
            tf.matmul(fc, weights["l3"]) + biases["l3"])

        fc = tf.nn.sigmoid(
            tf.matmul(fc, weights["l4"]) + biases["l4"])

        out_time_bin.append(fc)

    c = tf.concat(out_time_bin, 1)
    lstm_layer = tf.contrib.rnn.BasicLSTMCell(nodes["l5"], forget_bias=1)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layer, [c], dtype=tf.float32)
    prediction = tf.matmul( outputs[-1], weight([nodes["l5"], NUM_CLASSES])) + bias(NUM_CLASSES)
    return prediction

class Data_handle(object):
    def __init__(self, norm=100):
        self.lines = self.lines()
        self.z_index = self.z_index()
        self.det = Det(PATH + 'data/km3net_115.det')
        self.NORM_FACTOR = float(norm) 

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

    def make_event(self, hits, split_dom=True):
        "Take aa_net hits and put them in cube numpy arrays"
        ts = []
        for hit in hits:
            ts.append(hit.t)
       
        t0 = min(ts)
        t1 = max(ts)
        dt = t1 - t0 

        tbin_size = 400
        num_tbins = np.int(np.ceil(dt / tbin_size))
        channels = 3 if split_dom else 1
        event = np.zeros((num_tbins, 13, 13, 18, channels))


        for hit in hits:
            tot = hit.tot
            t = hit.t - t0
            
            t_index = np.int(np.floor(t / tbin_size))

            channel_id = hit.channel_id if split_dom else 0
            pmt = self.det.get_pmt(hit.dom_id, channel_id)
            direction = np.array([pmt.dir.x, pmt.dir.y, pmt.dir.z])
            dom = self.det.get_dom(pmt)
            line_id = dom.line_id
            # also valid
            # line_id = np.ceil(hit.dom_id / 18.)

            z = self.z_index[round(dom.pos.z)] 
            y, x = self.line_to_index(dom.line_id)
            try:
                event[t_index, x, y, z] += direction * tot / self.NORM_FACTOR 
            except IndexError:
                event[t_index - 1, x, y, z] += direction * tot / self.NORM_FACTOR 
                
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

def animate_event(event_full):
    """Shows 3D plot of evt"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    event_full = event_full.reshape((-1,13,13,18))

    ims = []
    for event in event_full:
        x, y, z = event.nonzero()
        k = event[event.nonzero()]
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

f = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')
def batches(batch_size):
    indices = np.random.choice(NUM_GOOD_TRAIN_EVENTS_3, NUM_GOOD_TRAIN_EVENTS_3, replace=False)
    for k in range(0, NUM_GOOD_TRAIN_EVENTS_3, 100):
        batch = indices[k: k + batch_size]
        events = np.zeros((batch_size, 50, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            tots, bins = f['all_tots'][j], f['all_bins'][j]
            E = f['all_energies'][j]
            x, y, z = f['all_positions'][j]
            dx, dy, dz = f['all_directions'][j]

            b = tuple(bins)
            events[i][b] = tots
            labels[i] = [E, x, y, z, dx, dy, dz] 
        yield events, labels

def train_batches(batch_size):
    indices = np.random.choice(range(NUM_GOOD_TRAIN_EVENTS_3, NUM_GOOD_EVENTS_3), NUM_GOOD_TEST_EVENTS_3, replace=False)
    for k in range(0, NUM_GOOD_TEST_EVENTS_3, 100):
        batch = indices[k: k + batch_size]
        events = np.zeros((batch_size, 50, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            tots, bins = f['all_tots'][j], f['all_bins'][j]
            E = f['all_energies'][j]
            x, y, z = f['all_positions'][j]
            dx, dy, dz = f['all_directions'][j]

            b = tuple(bins)
            events[i][b] = tots
            labels[i] = [E, x, y, z, dx, dy, dz] 
        yield events, labels

if __name__ == "__main__":
        pass
