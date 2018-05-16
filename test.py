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

for root_file, _ in root_files(train=False, test=True):
	root_part = root_file.split('/')[-1]	
	old = '/user/postm/data/root_files/' + root_part

	f[root_file + 'labels'] = f[old + 'labels']
	del f[old + 'labels']

	f[root_file] = f[old]
	del f[old]
