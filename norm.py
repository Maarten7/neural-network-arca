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

dens = False
he, be = spectrum_elec = np.histogram(np.log10(energies[np.where(ll == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, bee = elec_as_elec = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 0) )]), bins=be, range=range, density=dens)
hem, bem = elec_as_muon = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 1) )]), bins=be, range=range, density=dens)
hek, bek = elec_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 2) )]), bins=be, range=range, density=dens)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(be[:-1], hee / he.astype(float), label='shower as shower', ls='steps')
ax1.plot(be[:-1], hem / he.astype(float), label='shower as track', ls='steps')
ax1.plot(be[:-1], hek / he.astype(float), label='shower as k40', ls='steps')
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probebility to classify')
ax1.set_xlabel('log E')
#################################################################################################################################
hm, bm = spectrum_muon = np.histogram(np.log10(energies[np.where(ll == 1)]), bins=40, density=dens)
range=(be.min(), be.max())
hme, bme = elec_as_elec = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 0) )]), bins=be, range=range, density=dens)
hmm, bmm = elec_as_muon = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 1) )]), bins=be, range=range, density=dens)
hmk, bmk = elec_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 2) )]), bins=be, range=range, density=dens)

ax2.plot(be[:-1], hmm / hm.astype(float), label='track as track', ls='steps')
ax2.plot(be[:-1], hme / hm.astype(float), label='track as shower', ls='steps')
ax2.plot(be[:-1], hmk / hm.astype(float), label='track as k40', ls='steps')
ax2.legend()
ax2.set_title('classification track')
ax2.set_ylabel('Probebility to classify')
ax2.set_xlabel('log E')
plt.show()

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
energies = data_file['all_num_hits'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
dens = False
he, be = spectrum_elec = np.histogram(np.log10(energies[np.where(ll == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, bee = elec_as_elec = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 0) )]), bins=be, range=range, density=dens)
hem, bem = elec_as_muon = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 1) )]), bins=be, range=range, density=dens)
hek, bek = elec_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 0) & (lt == 2) )]), bins=be, range=range, density=dens)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(be[:-1], hee / he.astype(float), label='shower as shower', ls='steps')
ax1.plot(be[:-1], hem / he.astype(float), label='shower as track', ls='steps')
ax1.plot(be[:-1], hek / he.astype(float), label='shower as k40', ls='steps')
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probebility to classify')
ax1.set_xlabel('log num hits')
#################################################################################################################################
hm, bm = spectrum_muon = np.histogram(np.log10(energies[np.where(ll == 1)]), bins=40, density=dens)
range=(be.min(), be.max())
hme, bme = elec_as_elec = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 0) )]), bins=be, range=range, density=dens)
hmm, bmm = elec_as_muon = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 1) )]), bins=be, range=range, density=dens)
hmk, bmk = elec_as_k40  = np.histogram(np.log10(energies[np.where( (ll == 1) & (lt == 2) )]), bins=be, range=range, density=dens)

ax2.plot(be[:-1], hmm / hm.astype(float), label='track as track', ls='steps')
ax2.plot(be[:-1], hme / hm.astype(float), label='track as shower', ls='steps')
ax2.plot(be[:-1], hmk / hm.astype(float), label='track as k40', ls='steps')
ax2.legend()
ax2.set_title('classification track')
ax2.set_label('Probebility to classify')
ax2.set_xlabel('log num hits')
plt.show()
if __name__ == '__main__':
    pass
