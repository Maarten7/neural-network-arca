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


def import_model(only_model=True):
    """ imports a python module from command line. 
        Also import debug mode default is False"""        
    model = sys.argv[1].replace('/', '.')[:-3]
    model = importlib.import_module(model)
    if only_model: return model
    try:
        debug = eval(sys.argv[2])
        return model, debug
    except IndexError:
        return model, False

def make_file_str(evt_type, i):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_JEW_km3_v4_{0}_{1}.evt.root'
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
    print tra
    print tes
    return tra, tes 


NUM_DEBUG_EVENTS = 3000
#num_events(5)
DIR_TRAIN_EVENTS = {'anueNC': 20050, 'numuCC': 36638, 'nueCC': 30856, 'anumuCC': 38461, 'anueCC': 30970, 'nueNC': 22074, 'nuK40': 41172, 'anuK40': 41172}
DIR_TEST_EVENTS =  {'anueNC': 5055, 'numuCC': 8893, 'nueCC': 7623, 'anumuCC': 9540, 'anueCC': 7578, 'nueNC': 5595, 'nuK40': 10293, 'anuK40': 10293}
NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_EVENTS = NUM_TRAIN_EVENTS + NUM_TEST_EVENTS
