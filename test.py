from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *
#from sklearn.metrics import confusion_matrix
title = 'three_classes_sum_tot'

f = h5py.File(PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title, 'a')

batch_size = 100
root_file, _ = root_files(train=False, test=True).next()

events, labels = f[root_file], f[root_file + 'labels']

for j in range(0, len(labels), batch_size):
    e = events[j: j + batch_size]
    l = labels[j: j + batch_size]
    print len(l)
