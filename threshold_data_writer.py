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

EventFile.read_timeslices = True
def data_writer(title):
    dh = Data_handle() 
    for root_file, evt_type in root_files(train=True, test=True):
        print root_file 
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True

        type_index = EVT_TYPES.index(evt_type)
        lenf = 0
        for evt in f:
            num_hits = len(evt.mc_hits)
            if num_hits > 3 or num_hits == 0:
                lenf += 1
            
        with h5py.File(title, "a") as hfile:
            shape = (lenf, 13, 13, 18,) 
            dset_e = hfile.create_dataset(root_file, dtype='float64', shape=shape)
            shape = (lenf, 3)
            dset_l = hfile.create_dataset(root_file + "labels", dtype='int64', shape=shape)        
            dset_p = hfile.create_dataset(root_file + "positions", dtype='float64', shape=shape) 
            dset_d = hfile.create_dataset(root_file + "directions", dtype='float64', shape=shape) 
            dset_E = hfile.create_dataset(root_file + "energies", dtype='float64', shape=(lenf,)) 
            dset_t = hfile.create_dataset(root_file + "types", dtype='int', shape=(lenf,)) 
            dset_h = hfile.create_dataset(root_file + "num_hits", dtype='int', shape=(lenf,)) 
            ####################################################    
            i = 0 
            for evt in f:
               num_hits = len(evt.mc_hits)
               if num_hits > 3 or num_hits == 0:
                    event = dh.make_event(evt.hits)
                    try:
                        label = dh.make_labels(evt_type)
                    except TypeError:
                        label = dh.make_label(E, dx, dy, dz)

                    dset_e[i] = event     
                    dset_l[i] = label 
                    dset_t[i] = type_index

                    if num_hits == 0:
                        dset_E[i] = 0  
                        dset_h[i] = 0 
                        dset_p[i] = [0, 0, 0]
                        dset_d[i] = [0, 0, 0]

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
            
data_writer(title=PATH + 'data/hdf5_files/threshold_events_and_labels_%s.hdf5' % title)
