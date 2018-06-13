# datawriter.py
# Maarten Post

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
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))

    for root_file, evt_type in root_files(debug=True):
        print root_file 
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True
        
        num_events = len(f)
        with h5py.File(title, "a") as hfile:

            shape = (num_events, )
            dset_t = hfile.create_dataset(root_file + 'tots', dtype=dtf, shape=shape)
            shape = (num_events, 5)
            dset_b = hfile.create_dataset(root_file + 'bins', dtype=dti, shape=shape)
            shape = (num_events, 3)
            dset_l = hfile.create_dataset(root_file + "labels", dtype='int64', shape=shape)        
            ####################################################    
            
            for i, evt in enumerate(f):
                if i % 250 == 0:
                    print '\t', float(i) / num_events, '% done'
                tots, bins = dh.make_event(evt.hits, split_dom=True)
                try:
                    label = dh.make_labels(evt_type)
                except TypeError:
                    label = dh.make_label(E, dx, dy, dz)
                dset_t[i] = tots 
                dset_b[i] = bins 
                dset_l[i] = label 

            ####################################################
                
data_writer(title=PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title)

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
            
#meta_data_writer(title=PATH + 'data/hdf5_files/meta_data.hdf5')
