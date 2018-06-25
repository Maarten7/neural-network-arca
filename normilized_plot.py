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

model = import_model()
title = model.title

pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/all_events_labels_meta_%s.hdf5' % title, 'r')
predictions = pred_file['all_test_predictions'].value
labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
ll = np.argmax(labels, axis=1)
lt = np.argmax(predictions, axis=1)
eq = np.equal(ll, lt)

def energy_distribution(split):
    energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
    return hist_fill(np.log10(energies), split_false=split)


def hist_fill(list_value, only_extreme_output=False, k40=False, split_false=False):
    eg, ef, mg, mf  = [], [], [], []
    high_output, low_output = True, True

    kg, kf = [], []

    efm, efk = [], []
    mfe, mfk = [], []
    kfe, kfm = [], []
    for i in range(len(predictions)):
        x = list_value[i] 
        correct_type = ll[i]
        predict_type = lt[i]
        pr = predictions[i][correct_type]

        if only_extreme_output: 
            high_output = 0.97 < pr < 1
            low_output = 0. < pr <.03
        if eq[i]: # correct prediciont
            if correct_type == 0 and high_output: # electon
                eg.append(x) 
            elif correct_type == 1 and high_output: # muon
                mg.append(x)
            elif correct_type == 2 and high_output: # k40 
                kg.append(x)
        if not eq[i]: # incorrect prediction
            if correct_type == 0 and low_output: # electron
                ef.append(x) 
                if predict_type == 1:
                    efm.append(x)
                elif predict_type == 2:
                    efk.append(x)
            elif correct_type == 1 and low_output: # electron 
                mf.append(x)
                if predict_type == 0:
                    mfe.append(x)
                elif predict_type == 2:
                    mfk.append(x)
            elif correct_type == 2 and high_output: # k40 
                kf.append(x)
                if predict_type == 0:
                    kfe.append(x)
                elif predict_type == 1:
                    kfm.append(x)

    if split_false and k40:
        return eg, efm, efk, mg, mfe, mfk, kg, kfe, kfm
    if split_false and not k40:
        return eg, efm, efk, mg, mfe, mfk
    if k40:
        return eg, ef, mg, mf, kg, kf
    if not k40:
        return eg, ef, mg, mf


def histogram(distribution, bins, split=True, xlabel = '', normed=False, domain=None): 
    if not xlabel: xlabel = distribution.__name__.split('_')[0]
    dist = distribution(split=split)
    he = np.histogram(energies[np.where(ll == 0)], bins=bins)[0]
    hm = np.histogram(energies[np.where(ll == 1)], bins=bins)[0]

    plot_list = []
    if len(dist) == 6 and not split: 
        eg, ef, mg, mf, kg, kf = dist
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        plot_list.append((ax3, kf, 'k40 false'))
        plot_list.append((ax3, kg, 'k40 correct'))
    if len(dist) == 4 and not split:
        eg, ef, mg, mf = dist
        fig, (ax1, ax2) = plt.subplots(1,2)
        plot_list.append((ax1, ef, 'electron false'))
        plot_list.append((ax2, mf, 'muon false'))    

    if len(dist) == 9 and split:
        eg, efm, efk, mg, mfe, mfk, kg, kfe, kfm = dist
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        plot_list.append((ax3, kfe, 'k40 as electron'))
        plot_list.append((ax3, kfm, 'k40 as muon'))
        plot_list.append((ax3, kg,  'k40 correct'))
        plot_list.append((ax1, efm, 'electron as muon'))
        plot_list.append((ax1, efk, 'electron as k40'))
        plot_list.append((ax2, mfe, 'muon as electron'))    
        plot_list.append((ax2, mfk, 'muon as k40'))    
    if len(dist) == 6 and split:
        eg, efm, efk, mg, mfe, mfk = dist
        fig, (ax1, ax2) = plt.subplots(1,2)
        plot_list.append((ax1, efm, 'electron as muon', he))
        plot_list.append((ax1, efk, 'electron as k40', he))
        plot_list.append((ax2, mfe, 'muon as electron', hm))    
        plot_list.append((ax2, mfk, 'muon as k40', hm))    
    plot_list.append((ax1, eg, 'electron correct', he))
    plot_list.append((ax2, mg, 'muon correct', hm))    

    summ = np.empty(40)
    for ax, data, label, h in reversed(plot_list):    
    
        n, bins = np.histogram(data, bins=bins, range=domain, normed=normed)#, label=label, histtype='step')
        n = n / h.astype(float)
        ax.plot(bins[:-1], n, label=label, ls='steps')
        summ += n
        ax.set_title(distribution.__name__ + ' ' + label.split()[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number events')
        ax.legend()
    print summ
    plt.show()
        

if __name__ == '__main__':
    histogram(energy_distribution, bins=40, xlabel='$\log(E)$')
