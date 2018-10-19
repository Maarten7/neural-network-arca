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

for i in range(1,  16):
    root_file = PATH + 'data/root_files/triggered/out_JTP_km3_v3_nuK40_%i.root' % i
    print root_file
    # TIMESLICE
    EventFile.read_timeslices = True
    f = EventFile(root_file)
    print len(f)
#
#    data = np.zeros((len(f),2), dtype=int)
#    
#    num_events = 0 
#    for evt in f:
#        num_hits = len(evt.mc_hits)
#        
#        data[num_events, 0] = int(num_events)
#    
#        num_events += 1
    # JDAQ
    EventFile.read_timeslices = False 
    f = EventFile(root_file)
    print len(f)
#    j = 0
#    for evt in f:
#        j += 1
#        #data[evt.trigger_counter, 1] = int(evt.trigger_counter)
#    print j
#np.savetxt('triggertest.txt', data, fmt='%i', delimiter='\t')

#good_triggered_events = []
#for row in data:
#    # getriggered
#    if row[1] > 3 and row[2] != 0:
#        good_triggered_events.append(1)
#    # not triggered
#    if row[1] > 3 and row[2] == 0:
#        good_triggered_events.append(0)
            
                


