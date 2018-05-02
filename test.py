from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

EventFile.read_timeslices = True
rf = root_files(train=True, test=True)
root_file, _ = rf.next()
print root_file
f = EventFile(root_file)
f.use_tree_index_for_mc_reading = True
fi = iter(f)
evt = fi.next()

print len(evt.mc_hits)
for hit in evt.mc_hits:
    print hit 

print len(evt.mc_trks)
for trk in evt.mc_trks:
    print str(trk)


while True:
    try:
        evt = fi.next()
    except StopIteration:
        break

print len(evt.mc_hits)
print len(evt.mc_trks)
