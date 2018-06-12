from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

title = 'three_classes_sum_tot'


f = h5py.File(title, "r") 


#dset = f.create_dataset('hoi', dtype='float64', shape=(100,))
#
#
#for i in range(19):
#    dset[i] = 1
