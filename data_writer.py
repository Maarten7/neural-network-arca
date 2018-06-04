""" datawriter.py 
    Maarten Post
    This program can convert root file data into hdf5 files
    that can be 'fed' to neural networks

    # The neural network model should have a Data_handle class
    # where it is defined how the root/aanet evt class is represented
    # (in numpy arrays) to the neural network. Here that Data_handle class
    # is imported out of the model and used for transforming all evt objects
    # to numpy arrays. Since when training the neural network each event is trained
    # upon multiple times this is more efficent than to convert while training
""" 
from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

# importing model title and Data_handle class
model = sys.argv[1].replace('/', '.')[:-3]
model = importlib.import_module(model)
title = model.title
Data_handle = model.Data_handle

EventFile.read_timeslices = True
def data_writer():
    """ converts neutrino events from root files to
        given format given by nn model to numpy arrays and stores them 
        as hdf5 files."""

    dh = Data_handle() 

    # Only non zero elements of events are stored.
    # To allow variable length numpy arrays to be stored
    # in hdf5 format as special dtype needs to be created
    dtf = h5py.special_dtype(vlen=np.dtype('float64'))
    dti = h5py.special_dtype(vlen=np.dtype('int'))

    htitle = PATH + 'data/hdf5_files/events_and_labels2_%s.hdf5' % title
    with h5py.File(htitle, "a") as hfile:
        for root_file, evt_type in root_files(train=True, test=True):
            print root_file 
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
           
            # for each root file a dataset in the hdf5 file is created
            # is can be usefull for batching training data
            dset_t = hfile.create_dataset(root_file + "tots", dtype=dtf, shape=(len(f), 1,))
            dset_b = hfile.create_dataset(root_file + "bins", dtype=dti, shape=(len(f), 5,))
            dset_l = hfile.create_dataset(root_file + "labels", dtype='int64', shape=(len(f), 3))
            ####################################################    
           
            # Converts aanet evt.hist array in to numpy arrays
            for i, evt in enumerate(f):
                tots, bins = dh.make_event(evt.hits)

                # Label are either one hot encoded classes or 
                # numerical values as Energy
                try:
                    label = dh.make_labels(evt_type)
                except TypeError:
                    label = dh.make_label(E, dx, dy, dz)

            dset_t[i] = tots 
            dset_b[i] = bins 
            dset_l[i] = label 
            ####################################################
                

def meta_data_writer(title):
    """ Writes meta data of neutrino events from root files to
        hdf5 files. This data consist of Energy, Position of interaction,
        direction of neutrino and number of mc hits"""
    dh = Data_handle() 

    htitle = PATH + 'data/hdf5_files/meta_data.hdf5'
    with h5py.File(htitle, "w") as hfile:
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
                # Getting, E, pos, dir out of the evt objects
                # Since K40 type events don't have an mc_energy
                # (there is no neutrino) the energy (and direction
                # and position) are set to zero
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
            
data_writer()
#meta_data_writer()
