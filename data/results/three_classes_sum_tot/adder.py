from ROOT import *
import aa
import h5py
import numpy as np

from helper_functions import *

hfile = h5py.File('test_result_three_classes_sum_tot.hdf5', 'a')

num_events = len(hfile['labels'])
for i in range(num_events):
        hfile['JTrigger'][i] = 0 
#dset_T = hfile.create_dataset('threshold', dtype='int',     shape=(num_events,))
#dset_E = hfile.create_dataset('energies',  dtype='float64', shape=(num_events,))
#dset_N = hfile.create_dataset('num_hits',  dtype='int',     shape=(num_events,))
#dset_J = hfile.create_dataset('JTrigger',  dtype='int',     shape=(num_events,))

#EventFile.read_timeslices = True
i = 0
tot_events = 0
for j, (trigger_file, evt) in enumerate(trigger_files(train=False, test=True)):
    print trigger_file
    root_file = "JEW".join(trigger_file.split("JTP"))
    print root_file
    
    EventFile.read_timeslices = True
    w = EventFile(root_file)
    num_events = len(w)

    EventFile.read_timeslices = False 
    f = EventFile(trigger_file)
    for evt in f:
        z = evt.frame_index
        i = tot_events + z
        hfile['JTrigger'][i] = evt.trigger_mask
    tot_events += num_events

#EventFile.read_timeslices = True
#for j, (root_file, evt) in enumerate(root_files(train=False, test=True)):
#    print root_file
#
#    w = EventFile(root_file)
#    f.use_tree_index_for_mc_reading = True
#    num_events = len(f)
#    print num_events
#
#    for k, evt in enumerate(f):
#
#        num_hits = len(evt.mc_hits)
#
#        if 'K40' in root_file:
#            hfile['num_hits'][i] = 0 
#            hfile['energies'][i] = 0 
#            hfile['threshold'][i] = 1 
#
#        else:
#            hfile['num_hits'][i] = num_hits 
#            hfile['energies'][i] = evt.mc_trks[0].E 
#            hfile['threshold'][i] = doms_hit_pass_threshold(evt.mc_hits, threshold=5, pass_k40=True)
#           hfile['JTrigger'][i] = 

#        i += 1

hfile.close()

