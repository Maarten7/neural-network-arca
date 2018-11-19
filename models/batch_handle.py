import numpy as np
from toy_model import make_toy
from helper_functions import *

num_mini_timeslices = 200
EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3

def batches(batch_size, test=False, debug=False):
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    if debug:
        indices = np.random.choice(NUM_TRAIN_EVENTS, NUM_DEBUG_EVENTS, replace=False)
        num_events = NUM_DEBUG_EVENTS 
    elif test:
        indices = range(NUM_TRAIN_EVENTS, NUM_EVENTS)
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

        bins = [np.concatenate(bins[:,i]) for i in range(5)]
        bins.insert(0, extra_bin)
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        events[bins] = tots

        yield events, labels

def get_validation_set(validation_set_size=500):
    #f = h5py.File(PATH + 'data/hdf5_files/20000ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    indices = range(NUM_TRAIN_EVENTS, NUM_EVENTS)
    np.random.seed(0)
    indices = np.random.choice(indices, validation_set_size, replace=False)
    np.random.seed()
    batch = indices
    events = np.zeros((validation_set_size, num_mini_timeslices, 13, 13, 18, 3))
    labels = np.zeros((validation_set_size, NUM_CLASSES))
    for i, j in enumerate(batch):
        # get event bins and tots
        labels[i] = f['all_labels'][j]
        tots, bins = f['all_tots'][j], f['all_bins'][j]

        bins = tuple(bins)
        events[i][bins] = tots
    return events, labels

def toy_batches(batch_size, debug=False):
    while True:
        events = np.zeros((batch_size, num_mini_timeslices, 13, 13, 18, 3))
        labels = np.zeros((batch_size, 3))
        for i in range(batch_size):
            event, label = make_toy()
            events[i] = event
            labels[i] = label
        yield events, labels 

def get_toy_validation_set(size=500):
    np.random.seed(4)
    events, labels = toy_batches(size).next()
    np.random.seed()
    return events, labels

def see_slices(j, movie=True):
    f = h5py.File(PATH + 'data/hdf5_files/20000ns_100ns_all_events_labels_meta_%s.hdf5' % title, 'r')
    tots =  f['all_tots'][j]
    bins =  f['all_bins'][j]
    energy = f['all_energies'][j]
    evt_type = f['all_types'][j]
    num_hits = f['all_num_hits'][j]

    event = np.zeros((num_mini_timeslices, 13, 13, 18, 3))
    event[tuple(bins)] = tots

    if movie:
        animate_event(event) 
    else:
        event = np.sqrt(np.sum(np.square(event), axis=4))
        for minislice in event:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([0,13])
            ax.set_ylim([0,13])
            ax.set_zlim([0,18])
            ax.set_xlabel('x index')
            ax.set_ylabel('y index')
            ax.set_zlabel('z index')
            plt.title('TTOT on DOM E=%.1f  nh=%i  type=%s' % (energy, num_hits, EVT_TYPES[evt_type]))
            x, y, z = minislice.nonzero() 
            k = minislice[minislice.nonzero()]
            sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
            fig.colorbar(sc)
            plt.show()
    return 0

def animate_event(event_full):
    """Shows 3D plot of evt"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    event_full = np.sqrt(np.sum(np.square(event_full), axis=4))

    ims = []
    for i, event in enumerate(event_full):
        x, y, z = event.nonzero()
        k = event[event.nonzero()]
        print k
        sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
        ims.append([sc])
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_zlim([0,18])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_zlabel('z index')
    plt.title('TTOT on DOM')
    fig.colorbar(sc)
    ani = animation.ArtistAnimation(fig, ims)
    #writer = animation.writers['ffmpeg']
    plt.show()
