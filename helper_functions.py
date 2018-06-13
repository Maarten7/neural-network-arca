import datetime
import random
from ROOT import *
import aa
import numpy as np
import sys
import matplotlib
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy


matplotlib.rcParams.update({'font.size': 22})

PATH = "/user/postm/neural-network-arca/"
LOG_DIR = PATH + "log"
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3

def import_model():
    model = sys.argv[1].replace('/', '.')[:-3]
    model = importlib.import_module(model)
    return model

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

def make_file_str(evt_type, i):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_JTE_km3_v4_{0}_{1}.evt.root'
    n = path.format(evt_type, i)
    return n

def root_files(train=True, test=False, debug=False):
    trange = []
    if train: trange = range(1, 13)
    if test: trange += range(13,16) 
    if debug: trange = range(1, 4) 
    for i in trange:
        for evt_type in EVT_TYPES:
            n = make_file_str(evt_type, i)
            yield n, evt_type

def rotate_events(events):
    k = random.randint(0,3)
    return np.rot90(events, k=k, axis=(2,3)) 

def num_events(root_file_range):
    z = {}
    EventFile.read_timeslices = True
    for root_file, evt_type in root_file_range:
        f = EventFile(root_file)
        try:
            z[evt_type] += len(f)
        except KeyError:
            z[evt_type] = len(f)
    return z

def num_good_events(threshold):
    EventFile.read_timeslices = True
    for root_file, evt_type in root_files(test=True):
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True

        for evt in f:
            
            if len(evt.mc_hits) == 0 and evt_type != 'K40':
                print evt_type, root_file
                break
                


EventFile.read_timeslices = True
#eccf = '/user/postm/neural-network-arca/data/root_files/out_JTE_km3_v4_nueCC_1.evt.root'
mccf = '/user/postm/neural-network-arca/data/root_files/out_JTE_km3_v4_numuCC_1.evt.root'
#k40f = '/user/postm/neural-network-arca/data/root_files/out_JTE_km3_v4_nuK40_1.evt.root'
##
#f = EventFile(k40f)
#f.use_tree_index_for_mc_reading = True
#fi = iter(f)
#evt = fi.next()
#hit = event.hits[0]
#det = Det(PATH + 'data/km3net_115.det')
#pmt = det.get_pmt(hit.dom_id, hit.channel_id)
#dom = det.get_dom(pmt)
#lenf = len(f)
#f.set_index(lenf - 8)
#ecc1 = copy(f.evt)
#
#f = EventFile(mccf)
#f.use_tree_index_for_mc_reading = True
#f.set_index(3846)
#mcc1 = copy(f.evt)
#
#f = EventFile(k40f)
#f.use_tree_index_for_mc_reading = True
#lenf = len(f)
#f.set_index(lenf - 2)
#k401 = copy(f.evt)


DIR_TRAIN_EVENTS = {'e': 67755 + 83420, 'm': 96362, 'k': 82368}
DIR_TEST_EVENTS = {'e': 16970 + 20618, 'm': 23734, 'k': 20592}
DIR_GOOD_EVENTS_3 = {'e': 59906 + 83120, 'm': 99319}
NUM_DEBUG_EVENTS = 55227
NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_GOOD_EVENTS = sum(DIR_GOOD_EVENTS_3.values())
NUM_EVENTS = NUM_TRAIN_EVENTS+ NUM_TEST_EVENTS
