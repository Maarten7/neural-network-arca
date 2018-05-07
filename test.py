from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *
from sklearn.metrics import confusion_matrix
title = 'three_classes_sum_tot'

z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
q = h5py.File(PATH + 'data/hdf5_files/bg_file_%s.hdf5' % title  )
predictions = z['predictions_bg']
labels = z['labels_bg']
events = z['events_bg']

ll = np.argmax(labels, axis=1)
lt = np.argmax(predictions, axis=1)





