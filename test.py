from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

EventFile.read_timeslices = True
z = {}
for root_file, evt_type in root_files(train=True, test=True):
    print root_file,
    f = EventFile(root_file)
    f.use_tree_index_for_mc_reading = True
    try:     
        z[evt_type] += len(f) 

    except KeyError:
        z[evt_type] = len(f)

print z
