from ROOT import *
import aa
import numpy as np
import tensorflow as tf
import h5py
from helper_functions import *

PATH = "/user/postm/neural-network-arca/"
det = Det(PATH + 'data/km3net_115.det')

lines = np.array([
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
            [0,   0,   0,   0,   0,   0,   0,  100, 98,  96,  94,  92,  0 ]])


z_index =  {712: 0, 676: 1, 640: 2, 604: 3, 568: 4, 532: 5,
            496: 6, 460: 7, 424: 8, 388: 9, 352: 10, 316: 11, 
            280: 12,244: 13, 208: 14, 172: 15, 136: 16, 100: 17}

def pmt_id_to_dom_id(pmt_id):
    #channel_id = (pmt_id - 1) % 31
    dom_id     = (pmt_id - 1) / 31 + 1
    return dom_id

def line_to_index(line):
    """ Takes a line number and return it's
    position and it's slice. Where event are made like
    backslashes"""
    i, j = np.where(lines == line)
    return np.int(i), np.int(j)

def pmt_to_dom_index(pmt):
    dom  = det.get_dom(pmt)
    z    = z_index[round(dom.pos.z)] 
    y, x = line_to_index(dom.line_id)
    return x, y, z

def pmt_direction(pmt):
    """ returns direction vector of a PMT"""
    direction = pmt.dir
    return np.array([direction.x, direction.y, direction.z])

def hit_to_pmt(hit):
    return det.get_pmt(hit.dom_id, hit.channel_id)

def hit_time_to_index(hit, tbin_size):
    t = hit.t
    if   t > 20000: t_index = 20000 / tbin_size - 1
    elif t < 0:     t_index = 0
    else:           t_index = np.int(np.floor(t / tbin_size))
    return t_index

def make_event(hits, norm_factor=100, tot_mode=True, tbin_size=NUM_MINI_TIMESLICES):
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
        return np.array([1, 0, 0, 0])
    elif code < 6:
        return np.array([0, 1, 0, 0])
    elif code < 8:
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])
