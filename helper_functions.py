import datetime
import random
from ROOT import *
import aa
import numpy as np
import socket
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

host = socket.gethostname()

PATH = "/user/postm/neural-network-arca/"
LOG_DIR = PATH + "log"
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3

def show_event(event):
    """Shows 3D plot of evt"""
    assert np.shape(event) == (13, 13, 18)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = event.nonzero()
    k = event[event.nonzero()]
    sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Oranges'))
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_zlim([0,18])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_zlabel('z index')
    plt.title('TTOT on DOM')
    fig.colorbar(sc)
    plt.show()

def make_file_str(evt_type, i):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_JTE_km3_v4_{0}_{1}.evt.root'
    return path.format(evt_type, i)

def timestamp():
    return datetime.datetime.now().strftime("%m-%d_%H:%M:%S_")

def root_files(train=True, test=False, debug=False):
    trange = []
    if train: trange = range(1, 13)
    if test: trange += range(13,16) 
    if debug: trange = range(1, 3)
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

#EventFile.read_timeslices = True
#rf = root_files()
#rfile, _ = rf.next()
#print rfile
#f = EventFile(rfile)
#f.use_tree_index_for_mc_reading = True
#fi = iter(f)
#EVENT = fi.next()
#print EVENT

DIR_TRAIN_EVENTS = {'e': 67755 + 83420, 'm': 96362, 'k': 82368}
DIR_TEST_EVENTS = {'e': 16970 + 20618, 'm': 23734, 'k': 20592}
NUM_DEBUG_EVENTS = 55227
NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values()) #329905
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values()) #81914
NUM_EVENTS = NUM_TRAIN_EVENTS+ NUM_TEST_EVENTS #411819

DIR_GOOD_TRAIN_EVENTS_3 = {'anueNC': 26707, 'numuCC': 45231, 'nueCC': 38778, 'anumuCC': 47666, 'anueCC': 38898, 'nueNC': 29207, 'nuK40': 48048, 'anuK40': 48048} 
DIR_GOOD_TEST_EVENTS_3 = {'anueNC': 1906, 'numuCC': 3142, 'nueCC': 2803, 'anumuCC': 3280, 'anueCC': 2641, 'nueNC': 2086, 'nuK40': 3432, 'anuK40': 3432}
NUM_GOOD_TRAIN_EVENTS_3 = sum(DIR_GOOD_TRAIN_EVENTS_3.values())
NUM_GOOD_TEST_EVENTS_3 = sum(DIR_GOOD_TEST_EVENTS_3.values())
NUM_GOOD_EVENTS_3 = NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3 #242345 + 2* 48048 + 2 * 3432
