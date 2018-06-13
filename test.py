from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

title = 'temporal'

q = h5py.File('est.hdf5', 'r')

f = h5py.File(PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title, 'r')

z = f['user/postm/neural-network-arca/data/root_files/']


tt = q['all_tots']
bb = q['all_bins'] 
