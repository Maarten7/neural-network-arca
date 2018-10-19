# jtrigger.py
# Maarten Post
""" Takes the test data set and looks what events are triggered by JTriggerEfficenty
    in order to compare it with KM3NNET"""
from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *


j = 0
with h5py.File(PATH + 'data/hdf5_files/JTrigger_trigger.hdf5') as hfile:
    dset_pred = hfile.create_dataset('all_test_triggers', shape=(NUM_TEST_EVENTS,), dtype='int')
    for root_file, evt_type in root_files(train=False, test=True):
        print root_file
        
        # no k40 because has to be JTP
        if 'K40' in root_file:
            EventFile.read_timeslices = True
            f = EventFile(root_file)
            good_triggered_events = np.full(len(f), -1)
        else:

            # TIMESLICE
            EventFile.read_timeslices = True
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True

            data = np.zeros((len(f),3), dtype=int)
            for i, evt in enumerate(f):
                data[i, 0] = i  
                data[i, 1] = doms_hit_pass_threshold(evt.mc_hits, 5, False) 

            # JDAQ
            EventFile.read_timeslices = False
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True

            for evt in f:
                data[evt.trigger_counter, 2] = int(evt.trigger_counter)
            
            break

            # Compare JDAQ and TIMESLICES
            good_triggered_events = []
            for row in data:
                # getriggered
                if row[1] and row[2] != 0:
                    good_triggered_events.append(1)
                # not triggered
                if row[1] and row[2] == 0:
                    good_triggered_events.append(0)
            
        num_events = len(good_triggered_events)
        dset_pred[j: j + num_events] = good_triggered_events
        j = j + num_events 
