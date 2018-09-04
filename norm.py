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

"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"

pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/tbin400_all_events_labels_meta_%s.hdf5' % title, 'r')
predictions = pred_file['all_test_predictions'].value
labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
ll = np.argmax(labels, axis=1)
lt = np.argmax(predictions, axis=1)
eq = np.equal(ll, lt)

def plot_normelized_with_error(bins, tot_dis, par_dis, ax, label):
    error =  par_dis / tot_dis.astype(float)   * np.sqrt( 1./ par_dis + 1./tot_dis)
    ax.errorbar(bins[:-1], par_dis / tot_dis.astype(float), label=label, fmt='.', yerr=error) 
#############################################################################################################
# ENERGIE PLOT
#############################################################################################################
dens = False
energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
fig, (ax1, ax2) = plt.subplots(1,2)
################ shower
he, be = spectrum_elec = np.histogram(np.log10(energies[np.where(ll == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, bee = elec_as_elec = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 0) )]), bins=be, range=range, density=dens)
hem, bem = elec_as_muon = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 1) )]), bins=be, range=range, density=dens)
hek, bek = elec_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 2) )]), bins=be, range=range, density=dens)

plot_normelized_with_error(be, he, hee, ax1, label='shower as shower')
plot_normelized_with_error(be, he, hem, ax1, label='shower as track')
plot_normelized_with_error(be, he, hek, ax1, label='shower as k40')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probability to classify')
ax1.set_xlabel('log E')

################# track
hm, bm = spectrum_muon = np.histogram(np.log10(energies[np.where(ll == 1)]), bins=40, density=dens)
range=(bm.min(), bm.max())
hme, bme = muon_as_elec = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 0) )]), bins=bm, range=range, density=dens)
hmm, bmm = muon_as_muon = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 1) )]), bins=bm, range=range, density=dens)
hmk, bmk = muon_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 2) )]), bins=bm, range=range, density=dens)

plot_normelized_with_error(bm, hm, hmm, ax2, label='track as track')
plot_normelized_with_error(bm, hm, hme, ax2, label='track as shower')
plot_normelized_with_error(bm, hm, hmk, ax2, label='track as k40')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_ylabel('Probability to classify')
ax2.set_xlabel('log E')
#
plt.show()

#############################################################################################################
# N HIT PLOT 
#############################################################################################################
num_hits = data_file['all_num_hits'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
dens = False
fig, (ax1, ax2) = plt.subplots(1,2)
###################### shower
he, be = spectrum_elec = np.histogram(np.log10(num_hits[np.where(ll == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, bee = muon_as_elec = np.histogram(np.log10(num_hits[np.where( (ll == 0) & (lt == 0) )]), bins=be, range=range, density=dens)
hem, bem = muon_as_muon = np.histogram(np.log10(num_hits[np.where( (ll == 0) & (lt == 1) )]), bins=be, range=range, density=dens)
hek, bek = muon_as_k40  = np.histogram(np.log10(num_hits[np.where( (ll == 0) & (lt == 2) )]), bins=be, range=range, density=dens)

plot_normelized_with_error(be, he, hee, ax1, label='shower as shower')
plot_normelized_with_error(be, he, hem, ax1, label='shower as track')
plot_normelized_with_error(be, he, hek, ax1, label='shower as k40')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probability to classify')
ax1.set_xlabel('log num hits')
###################### track
hm, bm = spectrum_muon = np.histogram(np.log10(num_hits[np.where(ll == 1)]), bins=40, density=dens)
range=(bm.min(), bm.max())
hme, bme = elec_as_elec = np.histogram(np.log10(num_hits[np.where( (ll == 1) & (lt == 0) )]), bins=bm, range=range, density=dens)
hmm, bmm = elec_as_muon = np.histogram(np.log10(num_hits[np.where( (ll == 1) & (lt == 1) )]), bins=bm, range=range, density=dens)
hmk, bmk = elec_as_k40  = np.histogram(np.log10(num_hits[np.where( (ll == 1) & (lt == 2) )]), bins=bm, range=range, density=dens)

plot_normelized_with_error(bm, hm, hmm, ax2, label='track as track')
plot_normelized_with_error(bm, hm, hme, ax2, label='track as shower')
plot_normelized_with_error(bm, hm, hmk, ax2, label='track as k40')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_label('Probebility to classify')
ax2.set_xlabel('log num hits')
#
plt.show()
