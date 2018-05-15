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
            events = np.empty((0, 140, 13, 13, 18, 31))
            labels = np.empty((0, 3))
            
            for evt in f:
                hits = evt.hits
                event = dh.make_event(hits)
                print event.shape
                try:
                    label = dh.make_labels(evt_type)
                except TypeError:
                    label = dh.make_label(E, dx, dy, dz)

                
                events = np.append(events, [event], axis=0)
                labels = np.append(labels, [label], axis=0)
                    
            dset = hfile.create_dataset(root_file, data=events, dtype='float64')
            dset = hfile.create_dataset(root_file + "labels", data=labels, dtype='int64')        
            ####################################################
                

def meta_data_writer(title):
    dh = Data_handle() 
    with h5py.File(title, "w") as hfile:
        for root_file, evt_type in root_files(train=True, test=True):
            print root_file 
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            
            ####################################################    
            energies = np.empty((0))
            n_hits = np.empty((0)) 
            positions = np.empty((0,3))
            directions = np.empty((0,3))

            for evt in f:
                if evt_type is not "K40":
                    trk = evt.mc_trks[0] 
                    E = trk.E
                    pos = trk.pos
                    x, y, z = pos.x, pos.y, pos.z
                    dir = trk.dir
                    dx, dy, dz = dir.x, dir.y, dir.z
                else:
                    x, y, z = 0, 0, 0 
                    dx, dy, dz = 0, 0, 0 
                    E = 0

                energies = np.append(energies, E)
                n_hits = np.append(n_hits, len(evt.mc_hits))
                positions = np.append(positions, [[x, y, z]], axis=0)
                directions = np.append(directions, [[dx, dy, dz]], axis=0)
                    
            dset = hfile.create_dataset(root_file + "E", data=energies, dtype='int64')        
            dset = hfile.create_dataset(root_file + "n_hits", data=n_hits, dtype='int64')        
            dset = hfile.create_dataset(root_file + 'positions', data=positions)
            dset = hfile.create_dataset(root_file + 'directions', data=directions)
            ####################################################
            
#data_writer(title=PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title)
meta_data_writer(title=PATH + 'data/hdf5_files/meta_data.hdf5')
