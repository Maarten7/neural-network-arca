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
import sys
import h5py
import importlib
from helper_functions import *

model = import_model()
title = model.title
Data_handle = model.Data_handle

EventFile.read_timeslices = True
def data_writer(title):
    dh = Data_handle() 
    # these datatypes allow for variable length array to be saved into hdf5 format
    # this is needed since tots and bins differ per event
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))

    with h5py.File(title, "a") as hfile:
        # Data sets for data tots, bins, and label
        shape = (NUM_GOOD_EVENTS_3, )
        dset_t = hfile.create_dataset('all_tots', dtype=dtf, shape=shape)
        shape = (NUM_GOOD_EVENTS_3, 5)
        dset_b = hfile.create_dataset('all_bins', dtype=dti, shape=shape)
        shape = (NUM_GOOD_EVENTS_3, 3)
        dset_l = hfile.create_dataset("all_labels", dtype='int64', shape=shape)        

        # Data sets for meta data: Energy, n_hits, type, position and direction
        shape = (NUM_GOOD_EVENTS_3, )
        dset_E = hfile.create_dataset('all_energies', dtype='float64', shape=shape)
        dset_h = hfile.create_dataset('all_num_hits', dtype='int', shape=shape)
        dset_y = hfile.create_dataset('all_types', dtype='int', shape=shape)
        
        shape = (NUM_GOOD_EVENTS_3, 3)
        dset_p = hfile.create_dataset('all_positions', dtype='float64', shape=shape)
        dset_d = hfile.create_dataset('all_directions', dtype='float64', shape=shape)

        ####################################################    
        i = 0
        for root_file, evt_type in root_files(test=True):
            print root_file 
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            num_events = len(f)
            type_index = EVT_TYPES.index(evt_type)


            ####################################################    
           
            for j, evt in enumerate(f):
                # progress bar
                if j % 250 == 0:
                    print '\t', float(j) / num_events, '% done'

                # only events with more than 3 hits are saved since
                # 4 hits is needed at least to reconstrucd.
                num_hits = len(evt.mc_hits)
                if num_hits > 3 or num_hits == 0: 
                    # root hits transformed into numpy arrays. labels is made from 
                    # event type
                    tots, bins = dh.make_event(evt.hits, split_dom=True)
                    label = dh.make_labels(type_index)
                    
                    dset_t[i] = tots 
                    dset_b[i] = bins 
                    dset_l[i] = label 
                
                    dset_h[i] = 0 
                    dset_y[i] = type_index

                    # K40 have no energy, nhits, posistion and directions.
                    # all are set to zero
                    if num_hits == 0:
                        dset_E[i] = 0
                        dset_p[i] = [0, 0, 0]
                        dset_d[i] = [0, 0, 0] 

                    # Meta data set for neurtrino events
                    else:    
                        trk = evt.mc_trks[0]
                        dset_E[i] = trk.E
                        dset_h[i] = num_hits

                        pos = trk.pos
                        dset_p[i] = [pos.x, pos.y, pos.z]
                        dir = trk.dir
                        dset_d[i] = [dir.x, dir.y, dir.z]
                    
                    i += 1
            
            ####################################################
data_writer(PATH + 'data/hdf5_files/tbin50_all_events_labels_meta_%s.hdf5' % title)
