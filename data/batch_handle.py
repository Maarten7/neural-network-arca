import numpy as np
import h5py

from helper_functions import *

num_mini_timeslices = NUM_MINI_TIMESLICES

def batches(batch_size, test=False, debug=False):
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta.hdf5', 'r')
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta.hdf5', 'r')
    if debug:
        indices = np.random.choice(NUM_TRAIN_EVENTS, NUM_DEBUG_EVENTS, replace=False)
        num_events = NUM_DEBUG_EVENTS 
    elif test:
        f = h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5', 'r')
        indices = range(NUM_TEST_EVENTS)
        num_events = NUM_TEST_EVENTS
    else:
        indices = np.random.permutation(NUM_TRAIN_EVENTS)
        num_events = NUM_TRAIN_EVENTS

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
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        events[bins] = tots

        yield events, labels

def get_validation_set(validation_set_size=600, batch_size=15):
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5', 'r')
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta_test.hdf5', 'r')
    
    np.random.seed(0)
    for k in range(validation_set_size / batch_size):
        indices = range(0, NUM_TEST_EVENTS)
        indices = np.random.choice(indices, batch_size, replace=False)
        batch = indices
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        labels = np.zeros((batch_size, NUM_CLASSES))
        for i, j in enumerate(batch):
            # get event bins and tots
            labels[i] = f['all_labels'][j]
            tots, bins = f['all_tots'][j], f['all_bins'][j]

            bins = tuple(bins)
            events[i][bins] = tots

        yield events, labels

#def toy_batches(batch_size, debug=False):
#    while True:
#        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
#        labels = np.zeros((batch_size, 3))
#        for i in range(batch_size):
#            event, label = make_toy(num_mini_timeslices)
#            events[i] = event
#            labels[i] = label
#        yield events, labels 
#
#def get_toy_validation_set(size=100):
#    np.random.seed(4)
#    events, labels = toy_batches(size).next()
#    np.random.seed()
#    return events, labels
#
