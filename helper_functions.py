from ROOT import * 
import aa
import numpy as np
import sys
import importlib


PATH = "/user/postm/neural-network-arca/"
LOG_DIR = PATH + "log"

EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3
num_mini_timeslices = 50

def pmt_id_to_dom_id(pmt_id):
    #channel_id = (pmt_id - 1) % 31
    dom_id     = (pmt_id - 1) / 31 + 1
    return dom_id

def random_event(k40=False):
    """ return a random evt object from data set"""
    # random file by random number and evt type
    i         = np.random.randint(1,16)
    evt_types = EVT_TYPES if k40 else EVT_TYPES[:6] 
    evt_type  = np.random.choice(evt_types)
    
    # format path string
    path = PATH + 'data/root_files'
    path += '/out_JEW_km3_v4_{0}_{1}.evt.root'
    n = path.format(evt_type, i)
    print n

    # open with aa / root
    EventFile.read_timeslices = True 
    f = EventFile(n)
    f.use_tree_index_for_mc_reading = True

    # random evt from file
    index = np.random.randint(0, len(f))
    f.begin()
    f.set_index(index)
    print f.index
    return f.evt

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

def make_file_str(evt_type, i, J='JEW'):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_{2}_km3_v4_{0}_{1}.evt.root'
    n = path.format(evt_type, i, J)
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

def trigger_files(train=True, test=False, debug=False):
    trange = []
    if train: trange = range(1, 13)
    if test: trange += range(13,16) 
    if debug: trange = range(1, 4) 
    for i in trange:
        for evt_type in EVT_TYPES:
            n = make_file_str(evt_type, i, J='JTP')
            yield n, evt_type

def doms_hit_pass_threshold(mc_hits, threshold, pass_k40):
    """ checks if there a at least <<threshold>> doms
        hit by monte carlo hits. retuns true or false"""
    if len(mc_hits) == 0: return pass_k40 

    dom_id_set = set()
    for hit in mc_hits:
        dom_id = pmt_id_to_dom_id(hit.pmt_id)
        dom_id_set.add(dom_id)
        if len(dom_id_set) >= threshold:
            return True
    return False

def num_events(threshold):
    """ calculates number of events with 
        num dom hits> threshold"""
    tra = {typ: 0 for typ in EVT_TYPES}
    tes = {typ: 0 for typ in EVT_TYPES}
    EventFile.read_timeslices = True
        
    for root_file, evt_type in root_files(train=True, test=True):
        print root_file        
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True

        if threshold == 0:
            if any(num in root_file for num in ['13', '14', '15']):
                tes[evt_type] += len(f) 
            else:
                tra[evt_type] += len(f) 
            continue

        for evt in f:
            if doms_hit_pass_threshold(evt.mc_hits, threshold, pass_k40=True):
                if any(num in root_file for num in ['13', '14', '15']):
                    tes[evt_type] += 1
                else:
                    tra[evt_type] += 1
    print tra
    print tes
    return tra, tes 

train_data_file = PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta.hdf5'
test_data_file  = PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5'

DIR_TRAIN_EVENTS = {'anueNC': 17483, 'numuCC': 35222, 'nueCC': 28866, 'anumuCC': 36960, 'anueCC': 28990, 'nueNC': 19418, 'nuK40': 41172, 'anuK40': 41172}
DIR_TRAIN_LABELS = {'shower': DIR_TRAIN_EVENTS['nueCC'] + DIR_TRAIN_EVENTS['anueCC'] + DIR_TRAIN_EVENTS['nueNC'] + DIR_TRAIN_EVENTS['anueNC'],
                    'track':  DIR_TRAIN_EVENTS['numuCC'] + DIR_TRAIN_EVENTS['anumuCC'],
                    'k40':    DIR_TRAIN_EVENTS['nuK40'] + DIR_TRAIN_EVENTS['anuK40']}

DIR_TEST_EVENTS = {'anueNC': 4438, 'numuCC': 8593, 'nueCC': 7145, 'anumuCC': 9212, 'anueCC': 7124, 'nueNC': 4946, 'nuK40': 10293, 'anuK40': 10293}
DIR_TEST_LABELS = {'shower': DIR_TEST_EVENTS['nueCC'] + DIR_TEST_EVENTS['anueCC'] + DIR_TEST_EVENTS['nueNC'] + DIR_TEST_EVENTS['anueNC'],
                    'track':  DIR_TEST_EVENTS['numuCC'] + DIR_TEST_EVENTS['anumuCC'],
                    'k40':    DIR_TEST_EVENTS['nuK40'] + DIR_TEST_EVENTS['anuK40']}

NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_EVENTS = NUM_TRAIN_EVENTS + NUM_TEST_EVENTS
NUM_DEBUG_EVENTS = 3000

print "\n#Train Events, #Test Events"
print NUM_TRAIN_EVENTS
print NUM_TEST_EVENTS
print '\n'
print "Classes"
print DIR_TRAIN_LABELS
print DIR_TEST_LABELS
print '\n'
