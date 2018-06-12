# datawriter.py
# Maarten Post

from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

model = sys.argv[1].replace('/', '.')[:-3]
model = importlib.import_module(model)
title = model.title
Data_handle = model.Data_handle

def data_writer(title):
    EventFile.read_timeslices = True
    dh = Data_handle() 
    
    with h5py.File(title, "a") as hfile:
        shape = (NUM_GOOD_EVENTS_3, 13, 13, 18)
        dset_e = hfile.create_dataset('all_events', dtype='float64', shape=shape)
        shape = (NUM_GOOD_EVENTS_3, 3)
        dset_l = hfile.create_dataset("all_labels", dtype='int64', shape=shape)        

        shape = (NUM_GOOD_EVENTS_3,)
        dset_E = hfile.create_dataset('all_energies', dtype='float64', shape=shape)
        dset_h = hfile.create_dataset('all_num_hits', dtype='int', shape=shape)
        dset_t = hfile.create_dataset('all_types', dtype='int', shape=shape)
        shape = (NUM_GOOD_EVENTS_3, 3)
        dset_p = hfile.create_dataset('all_positions', dtype='float64', shape=shape)
        dset_d = hfile.create_dataset('all_directions', dtype='float64', shape=shape)

        for root_file, evt_type in root_files(train=True, test=True):
            print root_file 
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            
            type_index = EVT_TYPES.index(evt_type)
            ####################################################    
           
            i = 0
            for j, evt in enumerate(f):
                num_hits = len(evt.mc_hits)
                if num_hits > 3 or evt in ['nueK40', 'anueK40']:
                    event = dh.make_event(evt.hits)
                    try:
                        label = dh.make_labels(type_index)
                    except TypeError:
                        label = dh.make_label(E, dx, dy, dz)

                    dset_e[i] = event     
                    dset_l[i] = label 

                    trk = evt.mc_trks[0]
                    dset_E[i] = trk.E  
                    dset_h[i] = num_hits  
                    dset_t[i] = type_index
                    pos = trk.pos
                    dset_p[i] = [pos.x, pos.y, pos.z] 
                    dir = trk.dir
                    dset_d[i] = [dir.x, dir.y, dir.z] 

                    i += 1
            ####################################################
            
data_writer(title=PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title)
