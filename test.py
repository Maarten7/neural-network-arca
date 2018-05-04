from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *


EventFile.read_timeslices = True
rf = root_files()
root_file, _ = rf.next()
f = EventFile(root_file)
f.use_tree_index_for_mc_reading = True
fi = iter(f)
evt = fi.next()
