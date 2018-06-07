from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *
#from sklearn.metrics import confusion_matrix
title = 'temporal'
summ = 0
f = h5py.File(PATH + 'data/hdf5_files/events_and_labels2_%s.hdf5' % title, 'r')
for root_file, _ in root_files(debug=True):
    print root_file
    summ += len(f[root_file + 'labels'])

print summ
print summ / 500
print 500 * 4 * 2
