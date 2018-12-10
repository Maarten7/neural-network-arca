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
            ]
    assert file in files
    path = PATH + 'data/hdf5_files/%s.hdf5'
    return h5py.File(path % file, 'r')

def get_num_events(file):
    if 'test' in file:
        num_events = NUM_TEST_EVENTS
        indices = range(num_events)
    if 'no_threshold' in file:
        num_events = NUM_TEST_EVENTS_NO_TRESHOLD
        indices = range(num_events)
    else:
        num_events = NUM_TRAIN_EVENTS
        indices = np.random.permutation(num_events)
    return indices, num_events

def batches(batch_size, file):

    f = get_file(file)
    indices, num_events = get_num_events(file)

    for k in range(0, num_events, batch_size):
        if k + batch_size > num_events:
            batch_size = num_events - k

        batch = sorted(list(indices[k: k + batch_size]))
        
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


def get_validation_set(validation_set_size=600, batch_size=15):
    f = get_file(test=True)

    np.random.seed(0)
    for k in range(validation_set_size / batch_size):
        indices = range(0, NUM_TEST_EVENTS)
        indices = np.random.choice(indices, batch_size, replace=False)
        batch = indices
        events = np.zeros((batch_size, NUM_MINI_TIMESLICES, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            # get event bins and tots
            labels[i] = f['all_labels'][j]
            tots, bins = f['all_tots'][j], f['all_bins'][j]

            bins = tuple(bins)
            events[i][bins] = tots

        yield events, labels

def random_tots_bins(num_hits_threshold=0):
    f = get_file() 
    nh = 0
    while nh < num_hits_threshold:  
        i = np.random.randint(NUM_TRAIN_EVENTS)
        nh = f['all_num_hits'][i]
    tots = f['all_tots'][i]
    bins = f['all_bins'][i]
    E    = f['all_energies'][i]
    typ  = f['all_types'][i]
    return tots, bins, E, nh, i, typ

def index_tots_bins(i):
    f = get_file() 
    nh = f['all_num_hits'][i]
    tots = f['all_tots'][i]
    bins = f['all_bins'][i]
    E    = f['all_energies'][i]
    typ  = f['all_types'][i]
    return tots, bins, E, nh, i, typ
