from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

title = 'temporal'

f = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')

