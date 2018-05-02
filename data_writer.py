# datawriter.py
# Maarten Post

from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

model = sys.argv[1]
model = importlib.import_module(model)
title = model.title
Data_handle = model.Data_handle

EventFile.read_timeslices = True
def data_writer(title):
    dh = Data_handle() 
    with h5py.File(title, "w") as hfile:
        for root_file, evt_type in root_files(train=True, test=True):
            print root_file 
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            
            ####################################################    
            events = np.empty((0, 1, 13, 13, 18))
            labels = np.empty((0, 3))
            energies = np.empty((0))
            n_hits = np.empty((0)) 
            for evt in f:
                try: 
                    E = evt.mc_trks[0].E
                except IndexError:
                    E = 0
                        
                hits = evt.hits
                event = dh.make_event(hits)
                label = dh.make_labels(evt_type)

                
                events = np.append(events, [event], axis=0)
                labels = np.append(labels, [label], axis=0)
                energies = np.append(energies, E)
                n_hits = np.append(n_hits, len(evt.mc_hits))
                    
            dset = hfile.create_dataset(root_file, data=events, dtype='float64')
            dset = hfile.create_dataset(root_file + "labels", data=labels, dtype='int64')        
            dset = hfile.create_dataset(root_file + "E", data=energies, dtype='int64')        
            dset = hfile.create_dataset(root_file + "n_hits", data=n_hits, dtype='int64')        
            ####################################################
            
data_writer(title=PATH + 'data/hdf5_files/bg_file_%s.hdf5' % title)
