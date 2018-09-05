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

# hdf5 files met (meta)data
pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/tbin50_all_events_labels_meta_%s.hdf5' % title, 'r')
predictions = pred_file['all_test_predictions'].value

# alle informatie van alle events
labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
num_hits = data_file['all_num_hits'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
types = data_file['all_types'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
positions = data_file['all_positions'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
directions = data_file['all_directions'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]

l_true = np.argmax(labels, axis=1)
l_pred = np.argmax(predictions, axis=1)
eq = np.equal(l_true, l_pred)

def plot_normelized_with_error(bins, tot_dis, par_dis, ax, label):
    error =  par_dis / tot_dis.astype(float)   * np.sqrt( 1./ par_dis + 1./tot_dis)
    #ax.errorbar(bins[:-1], par_dis / tot_dis.astype(float), label=label, fmt='.', yerr=error) 
    ax.plot(bins[:-1], par_dis / tot_dis.astype(float), label=label, drawstyle='steps') 
    ax.fill_between(bins[:-1], par_dis / tot_dis.astype(float) - error, par_dis / tot_dis.astype(float) + error, alpha=0.2, step='pre') 

#########################################################################################################################################################
# Richting NEUTRINO IN DETECTOR ###################$###############################################################################################################
#########################################################################################################################################################
theta = np.arctan2(directions[:,2],np.sqrt(np.sum(directions[:,0:2]**2, axis=1)))
phi = np.arctan2(directions[:,1], directions[:,0]) 
plt.hist(np.cos(theta[np.where(l_true != 2)]), bins=30)
#plt.show()

exit()
#############################################################################################################
# ENERGIE PLOT
#############################################################################################################
dens = False
fig, (ax1, ax2) = plt.subplots(1,2)
################ shower
he, be = spectrum_elec = np.histogram(np.log10(energies[np.where(l_true == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, _ = np.histogram(np.log10(energies[np.where( (l_true == 0) & (l_pred == 0) )]), bins=be, range=range, density=dens)
hem, _ = np.histogram(np.log10(energies[np.where( (l_true == 0) & (l_pred == 1) )]), bins=be, range=range, density=dens)
hek, _ = np.histogram(np.log10(energies[np.where( (l_true == 0) & (l_pred == 2) )]), bins=be, range=range, density=dens)

plot_normelized_with_error(be, he, hee, ax1, label='as shower')
plot_normelized_with_error(be, he, hem, ax1, label='as track')
plot_normelized_with_error(be, he, hek, ax1, label='as k40')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probability to classify shower as')
ax1.set_xlabel('log E')

################# track
hm, bm = spectrum_muon = np.histogram(np.log10(energies[np.where(l_true == 1)]), bins=40, density=dens)
range=(bm.min(), bm.max())
hme, _ = np.histogram(np.log10(energies[np.where( (l_true == 1) & (l_pred == 0) )]), bins=bm, range=range, density=dens)
hmm, _ = np.histogram(np.log10(energies[np.where( (l_true == 1) & (l_pred == 1) )]), bins=bm, range=range, density=dens)
hmk, _ = np.histogram(np.log10(energies[np.where( (l_true == 1) & (l_pred == 2) )]), bins=bm, range=range, density=dens)

plot_normelized_with_error(bm, hm, hmm, ax2, label='as track')
plot_normelized_with_error(bm, hm, hme, ax2, label='as shower')
plot_normelized_with_error(bm, hm, hmk, ax2, label='as k40')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_ylabel('Probability to classify track as')
ax2.set_xlabel('log E')
#
#plt.savefig('energy_distribution.pdf')
#plt.show()
plt.close()

#############################################################################################################
# N HIT PLOT 
#############################################################################################################
fig, (ax1, ax2) = plt.subplots(1,2)
###################### shower
he, be = spectrum_elec = np.histogram(np.log10(num_hits[np.where(l_true == 0)]), bins=40, density=dens)
range=(be.min(), be.max())
hee, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 0) & (l_pred == 0) )]), bins=be, range=range, density=dens)
hem, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 0) & (l_pred == 1) )]), bins=be, range=range, density=dens)
hek, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 0) & (l_pred == 2) )]), bins=be, range=range, density=dens)

plot_normelized_with_error(be, he, hee, ax1, label='as shower')
plot_normelized_with_error(be, he, hem, ax1, label='as track')
plot_normelized_with_error(be, he, hek, ax1, label='as k40')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Probability to classify shower as')
ax1.set_xlabel('log num hits')
###################### track
hm, bm = spectrum_muon = np.histogram(np.log10(num_hits[np.where(l_true == 1)]), bins=40, density=dens)
range=(bm.min(), bm.max())
hme, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 1) & (l_pred == 0) )]), bins=bm, range=range, density=dens)
hmm, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 1) & (l_pred == 1) )]), bins=bm, range=range, density=dens)
hmk, _ = np.histogram(np.log10(num_hits[np.where( (l_true == 1) & (l_pred == 2) )]), bins=bm, range=range, density=dens)

plot_normelized_with_error(bm, hm, hmm, ax2, label='as track')
plot_normelized_with_error(bm, hm, hme, ax2, label='as shower')
plot_normelized_with_error(bm, hm, hmk, ax2, label='as k40')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_label('Probebility to classify track')
ax2.set_xlabel('log num hits')
#
#plt.savefig('num_hits_distribution.pdf')
#plt.show()
plt.close()



#############################################################################################################
# ENERGIE PLOT SPLIT FOR ALL TYPES
#############################################################################################################
#EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
fig, (ax1, ax2) = plt.subplots(1,2)

he_nueCC, be = np.histogram(np.log10(energies[np.where( types == 0)]), bins=40, density=dens)
he_nueCC_as_shower, _ = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueCC, be = np.histogram(np.log10(energies[np.where( types == 1)]), bins=40, density=dens)
he_anueCC_as_shower, _ = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_nueNC, be = np.histogram(np.log10(energies[np.where( types == 2)]), bins=40, density=dens)
he_nueNC_as_shower, _ = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueNC, be = np.histogram(np.log10(energies[np.where( types == 3)]), bins=40, density=dens)
he_anueNC_as_shower,_  = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 3) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_numCC, be = np.histogram(np.log10(energies[np.where( types == 4)]), bins=40, density=dens)
he_numCC_as_shower, _ = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 4) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anumCC, be = np.histogram(np.log10(energies[np.where( types == 5)]), bins=40, density=dens)
he_anumCC_as_shower, _ = np.histogram(np.log10(energies[np.where( (l_pred == 0) & (types == 5) )]), bins=be, range=(be.min(), be.max()), density=dens)


plot_normelized_with_error(be, he_nueCC, he_nueCC_as_shower, ax1, label='elec CC')
plot_normelized_with_error(be, he_anueCC, he_anueCC_as_shower, ax1, label='a elec CC')
plot_normelized_with_error(be, he_nueNC, he_nueNC_as_shower, ax1, label='elec NC' )
plot_normelized_with_error(be, he_anueNC, he_anueNC_as_shower, ax1, label='a elec NC')
plot_normelized_with_error(be, he_numCC, he_numCC_as_shower, ax1, label='muon CC')
plot_normelized_with_error(be, he_anumCC, he_anumCC_as_shower, ax1, label='a muon CC')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Fraction of events classified as shower')
ax1.set_xlabel('log E')
########### track

he_nueCC, be = np.histogram(np.log10(energies[np.where( types == 0)]), bins=40, density=dens)
he_nueCC_as_track, _ = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueCC, be = np.histogram(np.log10(energies[np.where( types == 1)]), bins=40, density=dens)
he_anueCC_as_track, _ = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_nueNC, be = np.histogram(np.log10(energies[np.where( types == 2)]), bins=40, density=dens)
he_nueNC_as_track, _ = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueNC, be = np.histogram(np.log10(energies[np.where( types == 3)]), bins=40, density=dens)
he_anueNC_as_track,_  = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 3) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_numCC, be = np.histogram(np.log10(energies[np.where( types == 4)]), bins=40, density=dens)
he_numCC_as_track, _ = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 4) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anumCC, be = np.histogram(np.log10(energies[np.where( types == 5)]), bins=40, density=dens)
he_anumCC_as_track, _ = np.histogram(np.log10(energies[np.where( (l_pred == 1) & (types == 5) )]), bins=be, range=(be.min(), be.max()), density=dens)


plot_normelized_with_error(be, he_nueCC, he_nueCC_as_track, ax2, label='elec CC')
plot_normelized_with_error(be, he_anueCC, he_anueCC_as_track, ax2, label='a elec CC')
plot_normelized_with_error(be, he_nueNC, he_nueNC_as_track, ax2, label='elec NC' )
plot_normelized_with_error(be, he_anueNC, he_anueNC_as_track, ax2, label='a elec NC')
plot_normelized_with_error(be, he_numCC, he_numCC_as_track, ax2, label='muon CC')
plot_normelized_with_error(be, he_anumCC, he_anumCC_as_track, ax2, label='a muon CC')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_ylabel('Fraction of events classified as track')
ax2.set_xlabel('log E')
#plt.show()
plt.close()


#############################################################################################################
# N HITS PLOT SPLIT FOR ALL TYPES
#############################################################################################################
#EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
types = data_file['all_types'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
fig, (ax1, ax2) = plt.subplots(1,2)

he_nueCC, be = np.histogram(np.log10(num_hits[np.where( types == 0)]), bins=40, density=dens)
he_nueCC_as_shower, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueCC, be = np.histogram(np.log10(num_hits[np.where( types == 1)]), bins=40, density=dens)
he_anueCC_as_shower, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_nueNC, be = np.histogram(np.log10(num_hits[np.where( types == 2)]), bins=40, density=dens)
he_nueNC_as_shower, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueNC, be = np.histogram(np.log10(num_hits[np.where( types == 3)]), bins=40, density=dens)
he_anueNC_as_shower,_  = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 3) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_numCC, be = np.histogram(np.log10(num_hits[np.where( types == 4)]), bins=40, density=dens)
he_numCC_as_shower, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 4) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anumCC, be = np.histogram(np.log10(num_hits[np.where( types == 5)]), bins=40, density=dens)
he_anumCC_as_shower, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 0) & (types == 5) )]), bins=be, range=(be.min(), be.max()), density=dens)


plot_normelized_with_error(be, he_nueCC, he_nueCC_as_shower, ax1, label='elec CC')
plot_normelized_with_error(be, he_anueCC, he_anueCC_as_shower, ax1, label='a elec CC')
plot_normelized_with_error(be, he_nueNC, he_nueNC_as_shower, ax1, label='elec NC' )
plot_normelized_with_error(be, he_anueNC, he_anueNC_as_shower, ax1, label='a elec NC')
plot_normelized_with_error(be, he_numCC, he_numCC_as_shower, ax1, label='muon CC')
plot_normelized_with_error(be, he_anumCC, he_anumCC_as_shower, ax1, label='a muon CC')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification shower')
ax1.set_ylabel('Fraction of events classified as shower')
ax1.set_xlabel('log num hits')
########### track

he_nueCC, be = np.histogram(np.log10(num_hits[np.where( types == 0)]), bins=40, density=dens)
he_nueCC_as_track, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueCC, be = np.histogram(np.log10(num_hits[np.where( types == 1)]), bins=40, density=dens)
he_anueCC_as_track, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_nueNC, be = np.histogram(np.log10(num_hits[np.where( types == 2)]), bins=40, density=dens)
he_nueNC_as_track, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anueNC, be = np.histogram(np.log10(num_hits[np.where( types == 3)]), bins=40, density=dens)
he_anueNC_as_track,_  = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 3) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_numCC, be = np.histogram(np.log10(num_hits[np.where( types == 4)]), bins=40, density=dens)
he_numCC_as_track, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 4) )]), bins=be, range=(be.min(), be.max()), density=dens)

he_anumCC, be = np.histogram(np.log10(num_hits[np.where( types == 5)]), bins=40, density=dens)
he_anumCC_as_track, _ = np.histogram(np.log10(num_hits[np.where( (l_pred == 1) & (types == 5) )]), bins=be, range=(be.min(), be.max()), density=dens)


plot_normelized_with_error(be, he_nueCC, he_nueCC_as_track, ax2, label='elec CC')
plot_normelized_with_error(be, he_anueCC, he_anueCC_as_track, ax2, label='a elec CC')
plot_normelized_with_error(be, he_nueNC, he_nueNC_as_track, ax2, label='elec NC' )
plot_normelized_with_error(be, he_anueNC, he_anueNC_as_track, ax2, label='a elec NC')
plot_normelized_with_error(be, he_numCC, he_numCC_as_track, ax2, label='muon CC')
plot_normelized_with_error(be, he_anumCC, he_anumCC_as_track, ax2, label='a muon CC')
ax2.set_ylim(0,1)
ax2.legend()
ax2.set_title('classification track')
ax2.set_ylabel('Fraction of events classified as track')
ax2.set_xlabel('log num hits')
#plt.show()
plt.close()




#########################################################################################################################################################
# POSITIE IN DETECTOR ###################$###############################################################################################################
#########################################################################################################################################################
fig, (ax1, ax2) = plt.subplots(1,2)
afstand = np.sqrt(np.sum(positions ** 2, axis=1))

h, b  = np.histogram(np.log10(afstand[np.where( (l_true != 2) )]), bins=30, density=dens)
ha, _ = np.histogram(np.log10(afstand[np.where( (eq == True) &(l_true != 2) )]), bins=b, range=(b.min(), b.max()), density=dens)
hr, _ = np.histogram(np.log10(afstand[np.where( (eq == False)&(l_true != 2) )]), bins=b, range=(b.min(), b.max()), density=dens)

plot_normelized_with_error(b, h, ha, ax1, label='goed')
plot_normelized_with_error(b, h, hr, ax1, label='fout')
ax1.set_ylim(0,1)
ax1.legend()
ax1.set_title('classification')
ax1.set_ylabel('Fraction of events classified correct')
ax1.set_xlabel('log R')
#plt.show()
plt.close()


