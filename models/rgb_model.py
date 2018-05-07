#!/usr/bin/python -i
import tensorflow as tf
from ROOT import *
import aa
import numpy as np
from helper_functions import *

title = 'rgb'

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    # size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, 
        ksize=[1, 2, 2, 2, 1],
        strides=[1, 2, 2, 2, 1],
        padding='SAME')

def weights(shape):
    w = tf.Variable(tf.random_normal(shape=shape), name="Weights")
    return w


def bias(shape):
    b = tf.Variable(tf.random_normal(shape=[shape]), name="Bias")
    return b

def print_tensor(x):
    print 'x\t\t', x.shape, np.prod(x._shape_as_list()[1:])

x = tf.placeholder(tf.float32, [None, 31, 13, 13, 18], name="X_placeholder")
y = tf.placeholder(tf.float32, [None, 3], name="Y_placeholder")


def cnn(x):
    x = tf.reshape(x, shape=[-1, 13, 13, 18, 31]) 
    print_tensor(x)
    with tf.name_scope("Conv1"):
        nodes1 = 35 
        conv1 = tf.nn.relu(
            conv3d(x, weights([4, 4, 4, 31, nodes1])) + bias(nodes1))
        print_tensor(conv1)

    with tf.name_scope("Conv2"):
        nodes2 = 35 
        conv2 = tf.nn.relu(
            conv3d(conv1, weights([3, 3, 3, nodes1, nodes2])) + bias(nodes2))
        print_tensor(conv2)

        conv2 = maxpool3d(conv2)
        print_tensor(conv2)

    with tf.name_scope("Conv3"):
        nodes3 = 15 
        conv3 = tf.nn.relu(
            conv3d(conv2, weights([2, 2, 2, nodes2, nodes3])) + bias(nodes3))
        print_tensor(conv3)
	
        conv3 = maxpool3d(conv3)
        print_tensor(conv3)
    
    elements = np.prod(conv3._shape_as_list()[1:])
    fc = tf.reshape(conv3, [-1, elements])
    print_tensor(fc)
    with tf.name_scope("FullyC1"):
	nodes4 = 100_
        fc = tf.nn.sigmoid(
            tf.matmul(fc, weights([elements, 100])) + bias(100))
  	print_tensor(fc)

    with tf.name_scope("FullyC2"):
	nodes5 = 40
        fc = tf.nn.sigmoid(
            tf.matmul(fc, weights([nodes4, nodes5])) + bias(nodes5))
        print_tensor(fc)

        labels = 3
        output = tf.nn.softmax(tf.matmul(fc, weights([nodes5, labels])) + bias(labels))
        print_tensor(output)
    return output

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

    def make_event(self, hits):
        "Take aa_net hits and put them in cube numpy arrays"
        event = np.zeros((31, 13, 13, 18))
        for hit in hits:
            tot = hit.tot
	    channel_id = hit.channel_id
            pmt = self.det.get_pmt(hit.dom_id, channel_id)
            dom = self.det.get_dom(pmt)
            line_id = dom.line_id
            # also valid
            # line_id = np.ceil(hit.dom_id / 18.)

            z = self.z_index[round(dom.pos.z)] 
            y, x = self.line_to_index(dom.line_id)

            event[channel_id, x, y, z] += tot / self.NORM_FACTOR 
        return event

    def add_hit_to_event(self, event, hit):
        tot = hit.tot
        pmt = self.det.get_pmt(hit.dom_id, hit.channel_id)
        dom = self.det.get_dom(pmt)
        line_id = dom.line_id

        z = self.z_index[round(dom.pos.z)]
        y, x = self.line_to_index(dom.line_id)

        event[0, x, y, z] += tot / self.NORM_FACTOR
        return event
    
    def make_labels(self, code):
        """ Makes one hot labels form evt_type str"""
        if code == 'eCC' or code == 'eNC':
            return np.array([1, 0, 0])
        if code == 'mCC' or code == 'muCC':
            return np.array([0, 1, 0])
        if code == 'K40':
            return np.array([0, 0, 1])

EVT_TYPES = ['eCC', 'eNC', 'muCC', 'K40']

if __name__ == "__main__":
    cnn(x)
