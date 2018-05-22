import h5py 
import numpy as np
import importlib
import sys
import os
import importlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools
from helper_functions import *

model = sys.argv[1].replace('/', '.')[:-3]
try:
	model = importlib.import_module(model)
	title = getattr(model, "title") 
except ImportError:
	title = 'three_classes_sum_tot'

data_file = h5py.File(PATH + 'data/hdf5_files/events_and_labels_%s.hdf5' % title)
pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
meta_file = h5py.File(PATH + 'data/hdf5_files/meta_data.hdf5')
predictions = pred_file['predictions']
labels = pred_file['labels']
ll = np.argmax(labels.value, axis=1)
lt = np.argmax(predictions.value, axis=1)
eq = np.equal(ll, lt)

def hist_fill(list_value_x, list_value_y, only_extreme_output=False, k40=False, split_false=False):
    eg, ef, mg, mf  = [], [], [], []
    high_output, low_output = True, True

    kg, kf = [], []

    efm, efk = [], []
    mfe, mfk = [], []
    kfe, kfm = [], []
    for i in range(len(predictions)):
        x = list_value_x[i] 
        y = list_value_y[i]
        correct_type = ll[i]
        predict_type = lt[i]
        pr = predictions[i][correct_type]

        if only_extreme_output: 
            high_output = 0.97 < pr < 1
            low_output = 0. < pr <.03
        if eq[i]: # correct prediciont
            if correct_type == 0 and high_output: # electon
                eg.append((x, y)) 
            elif correct_type == 1 and high_output: # muon
                mg.append((x, y))
            elif correct_type == 2 and high_output: # k40 
                kg.append((x, y))
        if not eq[i]: # incorrect prediction
            if correct_type == 0 and low_output: # electron
                ef.append((x, y)) 
                if predict_type == 1:
                    efm.append((x, y))
                elif predict_type == 2:
                    efk.append((x, y))
            elif correct_type == 1 and low_output: # electron 
                mf.append((x, y))
                if predict_type == 0:
                    mfe.append((x, y))
                elif predict_type == 2:
                    mfk.append((x, y))
            elif correct_type == 2 and high_output: # k40 
                kf.append((x, y))
                if predict_type == 0:
                    kfe.append((x, y))
                elif predict_type == 1:
                    kfm.append((x, y))

    if split_false and k40:
        return eg, efm, efk, mg, mfe, mfk, kg, kfe, kfm
    if split_false and not k40:
        return eg, efm, efk, mg, mfe, mfk
    if k40:
        return eg, ef, mg, mf, kg, kf
    if not k40:
        return eg, ef, mg, mf

def energy_spectrum(bins):
    energies_e = []
    energies_m = []
    for root_file, evt_type in root_files(train=False, test=True):
        if evt_type == 'eCC' or evt_type == 'eNC':
            energies_e.extend(np.log(meta_file[root_file + 'E'].value))
        if evt_type == 'muCC':
            energies_m.extend(np.log(meta_file[root_file + 'E'].value))
    return np.histogram(energies_e, bins=bins), np.histogram(energies_m, bins=bins)

def nhits_distribution(split):
    nhits = []
    for root_file, evt_type in root_files(train=False, test=True):
        nhits.extend(meta_file[root_file + 'n_hits'].value)
    return hist_fill(nhits, split_false=split)

def phi_distribution(split):
    phis = []
    for root_file, evt_type in root_files(train=False, test=True):
        for dir in meta_file[root_file + 'directions'].value:
                dx, dy, dz = dir
                phi = np.arctan2(dy, dx) 
                phis.append(phi) 
    return hist_fill(phis, split_false=split) 

def energy_theta_distribution(split):
    energies = []
    for root_file, evt_type in root_files(train=False, test=True):
        energies.extend(np.log(meta_file[root_file + 'E'].value))
    thetas = []
    for root_file, evt_type in root_files(train=False, test=True):
        for dir in meta_file[root_file + 'directions'].value:
                dx, dy, dz = dir
                theta = np.arctan2(dz, math.sqrt(dx**2 + dy**2)) 
                thetas.append(np.cos(theta))
    return hist_fill(energies, thetas, split_false=split)
    

if __name__ == '__main__':
    out = energy_theta_distribution(split=True)
    out = [zip(*l) for l in out] 
        
    fig, axes = plt.subplots(2,3)
    
    for (x, y), ax in zip(out, axes.reshape(6)):
        ax.hist2d(x, y)
    plt.show()



