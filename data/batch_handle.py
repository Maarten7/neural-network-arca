import numpy as np
import h5py

from helper_functions import *

def get_file(file):
    files = [
            '20000ns_100ns_all_events_labels_meta',
            '20000ns_250ns_all_events_labels_meta',
            '20000ns_250ns_all_events_labels_meta_test',
            '20000ns_400ns_all_events_labels_meta',
            '20000ns_400ns_all_events_labels_meta_test',
            '20000ns_400ns_all_events_labels_meta_test_no_threshold',
            'all_400ns_with_ATM',
            'all_400ns_with_ATM_test',
            'all_400ns_with_ATM_validation',
            ]
    assert file in files
    path = PATH + 'data/hdf5_files/%s.hdf5'
    return h5py.File(path % file, 'r')

def get_num_events(file):
    if 'no_threshold' in file:
        num_events = NUM_TEST_EVENTS_NO_TRESHOLD
    elif 'test' in file:
        num_events = NUM_TEST_EVENTS
    elif 'validation' in file:
        num_events = NUM_VAL_EVENTS
    else:
        num_events = NUM_TRAIN_EVENTS
    return num_events

def batches(file, batch_size):

    f = get_file(file)
    num_events = get_num_events(file)

    for k in range(0, num_events, batch_size):
        if k + batch_size > num_events:
            batch_size = num_events % k

        batch = range(k, k + batch_size)
        
        tots = f['all_tots'][batch]
        bins = f['all_bins'][batch]
        labels = f['all_labels'][batch]

        num_hits = [len(tot) for tot in tots]
        extra_bin = np.concatenate([np.full(shape=j, fill_value=i) for i, j in zip(range(batch_size), num_hits)])
        tots = np.concatenate(tots)

        # np.vstack !!!
        bins = [np.concatenate(bins[:,i]) for i in range(5)]
        bins.insert(0, extra_bin)
        events = np.zeros((batch_size, NUM_MINI_TIMESLICES, 13, 13, 18, 3))
        events[bins] = tots

        yield events, labels

def random_tots_bins(num_hits_threshold=0):
    f = get_file('20000ns_400ns_all_events_labels_meta')
    nh = 0
    i = np.random.randint(NUM_TRAIN_EVENTS)
    while nh < num_hits_threshold:  
        i = np.random.randint(NUM_TRAIN_EVENTS)
        nh = f['all_num_hits'][i]
    tots = f['all_tots'][i]
    bins = f['all_bins'][i]
    E    = f['all_energies'][i]
    typ  = f['all_types'][i]
    f.close()
    return tots, bins, E, nh, i, typ

def index_tots_bins(i):
    f = get_file() 
    nh = f['all_num_hits'][i]
    tots = f['all_tots'][i]
    bins = f['all_bins'][i]
    E    = f['all_energies'][i]
    typ  = f['all_types'][i]
    f.close()
    return tots, bins, E, nh, i, typ

