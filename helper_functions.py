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

def save_output(cost, acc=0, epoch=0):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file"""

    print "Epoch %s\tcost: %f\tacc: %f" % (epoch, cost, acc)

    with open(PATH + 'data/results/%s/epoch_cost_acc.txt' % (model.title), 'a') as f:
        f.write(str(epoch) + ',' + str(cost) + ',' str(acc) + '\n')

def import_model():
    """ imports a python module from command line. 
        Also import debug mode default is False"""        
    model = sys.argv[1].replace('/', '.')[:-3]
    model = importlib.import_module(model)
    try:
        debug = eval(sys.argv[2])
        return model, debug
    except IndexError:
        return model, False

def make_file_str(evt_type, i):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_JTE_km3_v4_{0}_{1}.evt.root'
    n = path.format(evt_type, i)
    return n

def root_files(train=True, test=False, debug=False):
    """ outputs strings of all root_files"""
    trange = []
    if train: trange = range(1, 13)
    if test: trange += range(13,16) 
    if debug: trange = range(1, 4) 
    for i in trange:
        for evt_type in EVT_TYPES:
            n = make_file_str(evt_type, i)
            yield n, evt_type

def num_events(threshold):
    """ calculates number of events with 
        num_mc_hits > threshold"""
    tra = {typ: 0 for typ in EVT_TYPES}
    tes = {typ: 0 for typ in EVT_TYPES}
    EventFile.read_timeslices = True
    for root_file, evt_type in root_files(test=True):
        print root_file        
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True

        if 'K40' in evt_type:
            if any(num in root_file for num in ['13', '14', '15']):
                tes[evt_type] += len(f) 
            else:
                tra[evt_type] += len(f) 
            continue

        for evt in f:
            if len(evt.mc_hits) > threshold:
                if any(num in root_file for num in ['13', '14', '15']):
                    tes[evt_type] += 1
                else:
                    tra[evt_type] += 1
                    
    return tra, tes 


EventFile.read_timeslices = True
eccf = '/user/postm/neural-network-arca/data/root_files/out_JTE_km3_v4_nueCC_5.evt.root'
f = EventFile(eccf)
f.use_tree_index_for_mc_reading = True

DIR_TRAIN_EVENTS = {'e': 67755 + 83420, 'm': 96362, 'k': 82368}
DIR_TEST_EVENTS = {'e': 16970 + 20618, 'm': 23734, 'k': 20592}
NUM_DEBUG_EVENTS = 3000 # first it was 55227
NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_EVENTS = NUM_TRAIN_EVENTS + NUM_TEST_EVENTS


DIR_GOOD_TRAIN_EVENTS_3 = {'anueNC': 22866, 'numuCC': 38886, 'nueCC': 33337, 'anumuCC': 40807, 'anueCC': 33399, 'nueNC': 24971, 'nuK40': 41184, 'anuK40': 41184}
DIR_GOOD_TEST_EVENTS_3 = {'anueNC': 5747, 'numuCC': 9487, 'nueCC': 8244, 'anumuCC': 10139, 'anueCC': 8140, 'nueNC': 6322, 'nuK40': 10296, 'anuK40': 10296}

NUM_GOOD_TRAIN_EVENTS_3 = sum(DIR_GOOD_TRAIN_EVENTS_3.values())
NUM_GOOD_TEST_EVENTS_3 = sum(DIR_GOOD_TEST_EVENTS_3.values())
NUM_GOOD_EVENTS_3 = NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3
