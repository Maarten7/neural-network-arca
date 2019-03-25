# datawriter.py
# Maarten Post
""" This program takes all the root files and transforms the events
    into numpy array that can be used for training the model.
    
    For efficient storing only the non zero elements and corresponding positions
    of the (t, x, y, z, rbg) = (~28, 13, 13, 18, 3) shaped 4D matrixes are stored. 
    Durring training these need to put back into a large narray of zerro's.

    The nonzero elements contain the time over threshold (tots) of the doms and the positions
    are called the bins. Also the corresponding "one-hot" label is saved. The meta data of the event
    is also saved. These are: the energy, position, direction of the neutrino and the number of hits
    it make on the doms.
"""
from ROOT import * 
import aa
import numpy as np
import h5py
from time import time

from helper_functions import *
from data.detector_handle import make_event, make_labels
from data.batch_handle import random_tots_bins


def random_index_gen(num_events, test):
    if test:
        for i in range(num_events):
            yield i
    else:
        indices = np.random.permutation(num_events)
        for i in indices:
            yield i

def get_num_events(test=False):
    if test:
        return NUM_TEST_EVENTS
    else:
        return NUM_TRAIN_EVENTS


EventFile.read_timeslices = True
def data_writer(file, tbin_size):
    # these datatypes allow for variable length array to be saved into hdf5 format
    # this is needed since tots and bins differ per event
    test = 'test' in file 
    num_events = get_num_events(test) 
    root_files_gen = root_files_test if test else root_files_train
    
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))

    with h5py.File(file, "a") as hfile:
        # Data sets for data tots, bins, and label
        shape = (num_events, )
        dset_t = hfile.create_dataset('all_tots', dtype=dtf, shape=shape)

        shape = (num_events, 5)
        dset_b = hfile.create_dataset('all_bins', dtype=dti, shape=shape)

        shape = (num_events, NUM_CLASSES)
        dset_l = hfile.create_dataset("all_labels", dtype='int64', shape=shape)        

        ####################################################    

        random_i = random_index_gen(num_events, test)
        i = random_i.next() 

        for root_file, evt_type in root_files_gen():
            type_index = EVT_TYPES.index(evt_type)
            print root_file 

            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True

            ####################################################    
            for j, evt in enumerate(f):
                # only events with more than 3 hits are saved since
                # 4 hits is needed at least to reconstrucd.
                #if doms_hit_pass_threshold(evt.mc_hits, threshold=0, pass_k40=True): 
                if good_event(evt, evt_type):
                    # root hits transformed into numpy arrays. labels is made from 
                    # event type
                    ts = time()
                    tots, bins = make_event(evt.hits, tbin_size=tbin_size)
                    print float(time() - ts)
                    label = make_labels(type_index)
                    
                    dset_l[i] = label 
                    dset_t[i] = tots 
                    dset_b[i] = bins 
                    
                    i = random_i.next() 
                        
            
            ####################################################
def validation_file(file, test_file, num_events=NUM_VAL_EVENTS):
    # these datatypes allow for variable length array to be saved into hdf5 format # this is needed since tots and bins differ per event
    
    random_indices = np.random.choice(range(NUM_TEST_EVENTS), num_events, replace=False)
    
    test_file = h5py.File(test_file, 'r')
    
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))
    with h5py.File(file, "a") as hfile:
        # Data sets for data tots, bins, and label

        shape = (num_events, )
        dset_t = hfile.create_dataset('all_tots', dtype=dtf, shape=shape)

        shape = (num_events, 5)
        dset_b = hfile.create_dataset('all_bins', dtype=dti, shape=shape)

        shape = (num_events, NUM_CLASSES)
        dset_l = hfile.create_dataset("all_labels", dtype='int64', shape=shape)        

        for i, ri in enumerate(random_indices):
            dset_l[i] = test_file['all_labels'][ri]
            dset_b[i] = np.vstack(test_file['all_bins'][ri])
            dset_t[i] = test_file['all_tots'][ri]


    test_file.close()


def get_energy(evt, type_index):
    if type_index in [6,7]: return 0
    if type_index == 8: return np.sum([trk.E for trk in evt.mc_trks])
    return evt.mc_trks[0].E

def get_num_hits(evt, type_index):
    if type_index in [6, 7]: return len(evt.hits)
    return len(evt.mc_hits)

def get_num_muons(evt, type_index):
    if type_index != 8: return 0
    return len(evt.mc_trks) - 1

def get_muon_th(evt, type_index):
    return 0

def get_weigth(evt, type_index):
    if type_index in [6,7,8]: return 0
    return evt.w[2]

def get_dir(evt, type_index):
    if type_index in [6,7]: return 0
    if type_index == 8:
        d = evt.mc_trks[1].dir
        for trk in evt.mc_trks:
            print [trk.pos.x, trk.pos.y, trk.pos.z]
        print
    else:
        d = evt.mc_trks[0].dir
    return [d.x, d.y, d.z]

def get_thm(evt, type_index):
    if type_index != 8: return 0

    for trk in evt.mc_trks:
        t = trk
    hit_per_muon = np.zeros(t.id + 1)

    for hit in evt.mc_hits:
        hit_per_muon[hit.origin] += 1

    return (hit_per_muon > 5).sum()


def add_stuff(file):
    with h5py.File(file, 'a') as hfile:
        num_events = NUM_TEST_EVENTS
    
        #dset_l = hfile.create_dataset("all_labels", dtype=int, shape=(num_events,NUM_CLASSES))        
        #dset_e = hfile.create_dataset("all_energies", dtype=float, shape=(num_events,))        
        #dset_n = hfile.create_dataset("all_num_hits", dtype=int, shape=(num_events,))        
        #dset_m = hfile.create_dataset("all_num_muons", dtype=float, shape=(num_events,))        
        ##dset_m = hfile.create_dataset("all_masks", dtype=float, shape=(num_events,))        
        #dset_h = hfile.create_dataset("all_muon_th", dtype=float, shape=(num_events,))        
        #dset_w = hfile.create_dataset("all_weights", dtype=float, shape=(num_events,))        
        #dset_d = hfile.create_dataset("all_directions", dtype=float, shape=(num_events,3))        
        dset_t = hfile.create_dataset("all_types", dtype=float, shape=(num_events,))

        dset_l = hfile['all_labels']
        dset_e = hfile['all_energies']
        dset_n = hfile['all_num_hits']
        dset_m = hfile['all_num_muons']
        dset_w = hfile['all_weights']
        dset_d = hfile['all_directions']
        #dset_t = hfile['all_types']

        random_i = random_index_gen(num_events, test='test' in file)
        i = random_i.next() 
        for root_file, evt_type in root_files_test(): 
            type_index = EVT_TYPES.index(evt_type)
            print root_file, evt_type, type_index
            
            if type_index == 8:
                EventFile.read_timeslices = False
            else:
                EventFile.read_timeslices = True

            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            for evt in f:
                num_events = len(f)
                if good_event(evt, evt_type):
                    #dset_l[i] = make_labels(type_index)
                    #dset_e[i] = get_energy(evt, type_index)
                    #dset_n[i] = get_num_hits(evt, type_index) 
                    #dset_m[i] = get_num_muons(evt, type_index)
                    #dset_h[i] = get_muon_th(evt, type_index)
                    #dset_w[i] = get_weigth(evt, type_index) / float(num_events)
                    #dset_d[i] = get_dir(evt, type_index) 
                    #dset_h[i] = get_thm(evt, type_index)
                    dset_t[i] = type_index

                    i = random_i.next() 

#add_stuff(file='data/hdf5_files/test_file.hdf5')

data_writer(file='test.hdf5', tbin_size=400)
#data_writer(file='data/hdf5_files/400ns_ATM_BIG.hdf5', tbin_size=400)
#data_writer(file='data/hdf5_files/400ns_ATM_BIG_test.hdf5', tbin_size=400) 
#validation_file(file='data/hdf5_files/all_400ns_with_ATM_validation.hdf5', test_file='data/hdf5_files/all_400ns_with_ATM_test.hdf5')

