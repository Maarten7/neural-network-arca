import h5py 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools
#from helper_functions import *


matplotlib.rcParams.update({'font.size': 22})
dens = False
save = False 
"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"

# hdf5 files met (meta)data
data_file = h5py.File('test_result_three_classes_sum_tot.hdf5', 'r')


# Network output
threshold = data_file['threshold'].value
predictions = data_file['predictions'].value

# alle informatie van alle events
labels = data_file['labels'].value
energies = data_file['energies'].value
num_hits = data_file['num_hits'].value
trigger = data_file['JTrigger'].value

t = np.where(threshold == 1)
predictions = predictions[t]
labels = labels[t]
energies = energies[t]
num_hits = num_hits[t]
trigger = trigger[t]

# Predictions in to classes
l_true = np.argmax(labels, axis=1)
l_pred = np.argmax(predictions, axis=1)
eq = np.equal(l_true, l_pred)

energies[np.where( (l_true == 1) & (energies==0) )] = 10
num_hits[np.where( (l_true == 1) & (num_hits==0) )] = 1

#a = np.log10(pred_energy[np.where( l_true != 2 )])
#b = np.log10(energies[np.where( l_true != 2 )])
#plt.hist2d(a, b, bins=30, range=[[0, 1e8], [0, 1e8]])
#plt.show()

def plot_normelized_with_error(bins, tot_dis, par_dis, ax, label):
    error =  par_dis / tot_dis.astype(float)   * np.sqrt( 1./ par_dis + 1./tot_dis)
    #ax.errorbar(bins[:-1], par_dis / tot_dis.astype(float), label=label, fmt='.', yerr=error) 
    ax.plot(bins[:-1], par_dis / tot_dis.astype(float), label=label, drawstyle='steps') 
    ax.fill_between(bins[:-1], par_dis / tot_dis.astype(float) - error, par_dis / tot_dis.astype(float) + error, alpha=0.1, step='pre') 

def plot_confusion_matrix():
    cm = confusion_matrix(l_true, l_pred)
    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ))
    cm = (cm / summ) * 100
    err_cm = cm * np.sqrt( 1. / cm + 1. / summ) 

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(3)

    classes = ['shower', 'track', 'K40']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.1f} $\pm$ {1:.1f} %".format(cm[i,j], err_cm[i,j]), horizontalalignment='center', color='red')
    if save: 
        manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
        manager.resize(*manager.window.maxsize())
        #manager.frame.Maximize(True)
        plt.savefig('confusion_matrix' + '.pdf')

    plt.show()

def histogram_classified_as(data_histogram, xlabel, split=True):
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
    he, be  = np.histogram(data_histogram[np.where( (l_true == 1)                 & split)], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 0) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 1) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred == 2) & split)], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax2, label='as shower')
    plot_normelized_with_error(be, he, hem, ax2, label='as track')
    plot_normelized_with_error(be, he, hek, ax2, label='as K40')
    ax2.set_ylim(0,1)
    ax2.legend()
    ax2.set_title('Classification track')
    ax2.set_ylabel('Fraction of track events classified as')
    ax2.set_xlabel(xlabel)
    fig.set_size_inches(11.69, 8.27, forward=False)
    if save: 
        manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
        manager.resize(*manager.window.maxsize())
        #manager.frame.Maximize(True)
        fig.savefig('histogram_as_' + xlabel.replace(" ", "_") + '.pdf')
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
    fig.set_size_inches(11.69, 8.27, forward=False) 
    if save:
        fig.savefig('histogram_split_types_' + xlabel.replace(" ", "_") + '.pdf')
    plt.show()

def histogram_w(data_histogram, xlabel):
    fig, ax1 = plt.subplots(1,1)
    #### shower
    he, be  = np.histogram(data_histogram[np.where( (l_true != 2)                 )], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true != 2) & (l_pred != 2) )], bins=be, range=(be.min(), be.max()), density=dens)

    hjt, _  = np.histogram(data_histogram[np.where( (l_true != 2) & (trigger != 0) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax1, label='as event')
    plot_normelized_with_error(be, he, hjt, ax1, label='as event by trigger')
    ax1.set_ylim(0,1)
    ax1.legend()
    ax1.set_title('Classification events')
    ax1.set_ylabel('Fraction of events classified as')
    ax1.set_xlabel(xlabel)
    if save:
        fig.savefig('Jtrigger_' + xlabel.replace(" ", "_") + '.pdf')
    plt.show()

# all triggered events
def histogram_trigger(data_histogram, xlabel):
    fig, ax1 = plt.subplots(1,1)
    he, be  = np.histogram(data_histogram[np.where(                 (l_true != 2) )], bins=30, density=dens)
    hen, _  = np.histogram(data_histogram[np.where( (l_pred != 2)  &(l_true != 2) )], bins=be, range=(be.min(), be.max()), density=dens)
    het, _  = np.histogram(data_histogram[np.where( (triggers == 1)&(l_true != 2) )], bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(be, he, hen, ax1, label='NN')
    plot_normelized_with_error(be, he, het, ax1, label='JTrigger')
    ax1.set_ylim(0,1)
    ax1.legend()
    ax1.set_title('Trigger')
    ax1.set_ylabel('Fraction of events triggers correctly')
    ax1.set_xlabel(xlabel)
    if save:
        fig.savefig('trigger_' + xlabel.replace(" ", "_") + '.pdf')
    plt.show()

def events_triggerd_as_K40():
    print 'K3NNET  ', 100 * np.sum((l_pred != 2) & (l_true == 2)) / float(np.sum( l_true == 2)) 
    print 'JTrigger', 100 / float(np.sum( l_true == 2))

plot_confusion_matrix()

#histogram_classified_as(np.log10(energies), 'log E', Rxy < 250)
#histogram_classified_as(np.log10(energies), 'log E', ((250 < Rxy) & ( Rxy < 500)))
#histogram_classified_as(np.log10(energies), 'log E', Rxy > 500)
#histogram_classified_as(np.log10(energies), 'log E', outward)
#histogram_classified_as(np.log10(energies), 'log E', inward)

#histogram_classified_as(np.log10(energies), 'log E')
#histogram_classified_as(np.log10(num_hits), 'log N hits')

#histogram_classified_as(afstand, 'R meters')
#histogram_classified_as(np.log10(afstand), 'log R ')
#histogram_classified_as(np.cos(theta), 'cos theta')
#histogram_classified_as(theta, 'theta')
#histogram_classified_as(phi, 'phi')

#histogram_w(np.log10(energies), 'log E')
#histogram_w(np.log10(num_hits), 'log N hits')

#histogram_split_types(np.log10(energies), 'log E')
#histogram_split_types(np.log10(num_hits), 'log N hits')

#histogram_trigger(np.log10(energies), 'log E')
#histogram_trigger(np.log10(num_hits), 'log num hits')


#events_triggerd_as_K40()
