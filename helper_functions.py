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
PATH_RANCE = "/localstore/antares/Maarten_local/neural-network-arca/"
LOG_DIR = PATH + "log"
EVT_TYPES = ['eCC', 'eNC', 'muCC', 'K40']
NUM_CLASSES = 3

def show_event_3D(event):
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

def show_event(event):
    assert np.shape(event) == (13, 13, 18)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y, z = event.nonzero()
    sc = ax.scatter(x, y, zdir='z', cmap=plt.get_cmap('Oranges'))
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    fig.colorbar(sc)
    plt.show()

def make_file_str(evt_type, i, rance=False):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    if rance: path = PATH_RANCE + 'data/root_files'
    path += '/out_JTE_km3_v4_{0}{1}_{2}.evt.root'
    n = path.format('nu', evt_type, i)
    a = path.format('anu', evt_type, i)

    return n, a

def save_output(acc, cost, test=False):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file
        input test, boolean default False, if true the acc and cost are of the 
            test set"""
    mode = 'test' if test else 'train'
    with open(PATH + 'data/results/%s/acc_%s.txt' % (title, mode), 'a') as f:
        f.write(str(acc) + '\n')
    with open(PATH + 'data/results/%s/cost_%s.txt' % (title, mode), 'a') as f:
        f.write(str(epoch_loss / num_events) + '\n')

def timestamp():
    return datetime.datetime.now().strftime("%m-%d_%H:%M:%S_")

def root_files(train=True, test=False, debug=False, rance=False):
    trange = []
    if train: trange = range(1, 13)
    if test: trange += range(13,16) 
    if debug: trange = range(1, 3)
    for i in trange:
        for evt_type in EVT_TYPES:
            n, a = make_file_str(evt_type, i, rance=rance)
            yield n, evt_type
            yield a, evt_type

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

EventFile.read_timeslices = True
rf = root_files(rance=True)
rfile, _ = rf.next()
f = EventFile(rfile)
f.use_tree_index_for_mc_reading = True
fi = iter(f)
EVENT = fi.next()


DIR_TRAIN_EVENTS = {'e': 67755 + 83420, 'm': 96362, 'k': 82368}
DIR_TEST_EVENTS = {'e': 16970 + 20618, 'm': 23734, 'k': 20592}
NUM_DEBUG_EVENTS = 55227
NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_EVENTS = NUM_TRAIN_EVENTS+ NUM_TEST_EVENTS
