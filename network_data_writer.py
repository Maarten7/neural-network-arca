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

from helper_functions import *
from data.detector_handle import make_event, make_labels

def random_index_gen(num_events, test=False):
    if test:
        for i in range(num_events):
            yield i
    else:
        indices = np.random.permutation(num_events)
        for i in indices:
            yield i


EventFile.read_timeslices = True
def data_writer(title, tbin_size, range):
    # these datatypes allow for variable length array to be saved into hdf5 format
    # this is needed since tots and bins differ per event
    
    
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))

    with h5py.File(title, "a") as hfile:
        # Data sets for data tots, bins, and label
        shape = (num_events, )
        dset_t = hfile.create_dataset('all_tots', dtype=dtf, shape=shape)
        #shape = (num_events, 4)
        #dset_b = hfile.create_dataset('all_bins', dtype=dti, shape=shape)
        shape = (num_events, 5)
        dset_b = hfile.create_dataset('all_bins', dtype=dti, shape=shape)

        shape = (num_events, NUM_CLASSES)
        dset_l = hfile.create_dataset("all_labels", dtype='int64', shape=shape)        

        # Data sets for meta data: Energy, n_hits, type, position and direction
        shape = (num_events, )
        dset_E = hfile.create_dataset('all_energies', dtype='float64', shape=shape)
        dset_h = hfile.create_dataset('all_num_hits', dtype='int', shape=shape)
        dset_y = hfile.create_dataset('all_types', dtype='int', shape=shape)
        
        shape = (num_events, 3)
        dset_p = hfile.create_dataset('all_positions', dtype='float64', shape=shape)
        dset_d = hfile.create_dataset('all_directions', dtype='float64', shape=shape)

        ####################################################    

        random_i = random_index_gen(num_events, True)
        i = random_i.next() 

        for root_file, evt_type in root_files(range=range):
            type_index = EVT_TYPES.index(evt_type)
            print root_file, evt_type, type_index

            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            num_events = len(f)

            ####################################################    
            for j, evt in enumerate(f):
                # progress bar
                if j % 500 == 0:
                    print '\t%.2f %% done' % (float(j) / num_events)

                # only events with more than 3 hits are saved since
                # 4 hits is needed at least to reconstrucd.
                num_hits = len(evt.mc_hits)
                if doms_hit_pass_threshold(evt.mc_hits, threshold=0, pass_k40=True): 
                    # root hits transformed into numpy arrays. labels is made from 
                    # event type
                    tots, bins = make_event(evt.hits, tbin_size=tbin_size)
                    label = make_labels(type_index)
                    
                    dset_l[i] = label 
                    dset_t[i] = tots 
                    dset_b[i] = bins 
                
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

                        #label = make_labels(trk.E, pos.x, pos.y, pos.z, dir.x, dir.y, dir.z)
                    
                    i = random_i.next() 
            
            ####################################################
#data_writer(PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta.hdf5',      tbin_size=250, train=True, test=False)
data_writer(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test_no_threshold.hdf5', tbin_size=400, range=range(13, 16))
