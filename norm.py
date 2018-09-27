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

dens = False
"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"

# hdf5 files met (meta)data
pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (model.title, model.title), 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/tbin50_all_events_labels_meta_%s.hdf5' % model.title, 'r')

predictions = pred_file['all_test_predictions'].value
pred_energy = predictions[:, 0]
pred_direction = predictions[:,1:4]
pred_positions = predictions[:,4:7]

# alle informatie van alle events
labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
num_hits = data_file['all_num_hits'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
types = data_file['all_types'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
positions = data_file['all_positions'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
directions = data_file['all_directions'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]

l_true = np.argmax(labels, axis=1)
#l_pred = np.argmax(predictions, axis=1)
#eq = np.equal(l_true, l_pred)


a = np.log10(pred_energy[np.where( l_true != 2 )])
b = np.log10(energies[np.where( l_true != 2 )])

plt.hist2d(a, b, bins=30, range=[[0, 1e8], [0, 1e8]])
plt.show()

def plot_normelized_with_error(bins, tot_dis, par_dis, ax, label):
    error =  par_dis / tot_dis.astype(float)   * np.sqrt( 1./ par_dis + 1./tot_dis)
    #ax.errorbar(bins[:-1], par_dis / tot_dis.astype(float), label=label, fmt='.', yerr=error) 
    ax.plot(bins[:-1], par_dis / tot_dis.astype(float), label=label, drawstyle='steps') 
    ax.fill_between(bins[:-1], par_dis / tot_dis.astype(float) - error, par_dis / tot_dis.astype(float) + error, alpha=0.1, step='pre') 


def histogram_classified_as(data_histogram, xlabel):
    fig, (ax1, ax2) = plt.subplots(1,2)
    #### shower
    he, be  = np.histogram(data_histogram[np.where( (l_true == 0)                 )], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 0) )], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 1) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 2) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax1, label='as shower')
    plot_normelized_with_error(be, he, hem, ax1, label='as track')
    plot_normelized_with_error(be, he, hek, ax1, label='as K40')
    ax1.set_ylim(0,1)
    ax1.legend()
    ax1.set_title('Classification shower')
    ax1.set_ylabel('Fraction of shower events classified as')
    ax1.set_xlabel(xlabel)
    #### track
    he, be  = np.histogram(data_histogram[np.where( (l_true == 1)                 )], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 0) )], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 1) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 2) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax2, label='as shower')
    plot_normelized_with_error(be, he, hem, ax2, label='as track')
    plot_normelized_with_error(be, he, hek, ax2, label='as K40')
    ax2.set_ylim(0,1)
    ax2.legend()
    ax2.set_title('Classification track')
    ax2.set_ylabel('Fraction of track events classified as')
    ax2.set_xlabel(xlabel)
    plt.show()

def histogram_split_types(data, xlabel):
    fig, axes = plt.subplots(1,3)
    label_string = ['shower', 'track', 'K40']
    for j in range(0,3):
        for i in range(0,6):
            he, be = np.histogram(data[np.where( types == i)], bins=30, density=dens)
            he_as_type, _ = np.histogram(data[np.where( (l_pred == j) & (types == i) )], bins=be, range=(be.min(), be.max()), density=dens)
            plot_normelized_with_error(be, he, he_as_type, axes[j], label=EVT_TYPES[i])
        axes[j].set_ylim(0,1)
        axes[j].legend()
        axes[j].set_title(' ')
        axes[j].set_ylabel('Fraction of events classified as %s' % (label_string[j]))
        axes[j].set_xlabel(xlabel)
    plt.show()

#afstand = np.sqrt(np.sum(positions ** 2, axis=1))
#theta = np.arctan2(directions[:,2],np.sqrt(np.sum(directions[:,0:2]**2, axis=1)))
#phi = np.arctan2(directions[:,1], directions[:,0]) 

#histogram_classified_as(np.log10(energies), 'log E')
#histogram_classified_as(np.log10(num_hits), 'log N hits')
#histogram_classified_as(afstand, 'R meters')
#histogram_classified_as(np.log10(afstand), 'log R ')
#histogram_classified_as(np.cos(theta), 'cos theta')
#histogram_classified_as(theta, 'theta')
#histogram_classified_as(phi, 'phi')

#histogram_split_types(np.log10(energies), 'log E')
#histogram_split_types(np.log10(num_hits), 'log N hits')
