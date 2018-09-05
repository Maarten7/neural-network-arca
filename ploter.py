import h5py 
import numpy as np
import importlib
import sys
import os
import importlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import itertools
from helper_functions import *

from sklearn.metrics import confusion_matrix

model = import_model()
title = model.title

#pred_file = h5py.File(PATH + 'data/results/%s/test_result_vakantiepauze_%s.hdf5' % (title, title), 'r')
pred_file = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/tbin50_all_events_labels_meta_%s.hdf5' % title, 'r')
predictions = pred_file['all_test_predictions'].value
#labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
labels = data_file['all_labels'][NUM_GOOD_TRAIN_EVENTS_3 : NUM_GOOD_TRAIN_EVENTS_3 + len(predictions)]
ll = np.argmax(labels, axis=1)
lt = np.argmax(predictions, axis=1)
eq = np.equal(ll, lt)

def plot_confusion_matrix():
    cm = confusion_matrix(ll, lt)
    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ))
    cm = (cm / summ) * 100
    err_cm = cm * np.sqrt( 1. / cm + 1. / summ) 

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(3)

    classes = ['shower', 'track', 'K40']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.1f} $\pm$ {1:.1f} %".format(cm[i,j], err_cm[i,j]), horizontalalignment='center', color='red')

    plt.show()


def make_list_from_txt(title):
    file_handle = open(title, 'r')
    data = []
    for line in file_handle:
        line = line.split('_')
        try:
            value = float(line[-1])
            data.append(value)
        except ValueError:
            continue
    return data   

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

def nhits_distribution(split):
    nhits = data_file['all_num_hits'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
    return hist_fill(np.log10(nhits), split_false=split)

def energie_distribution(split):
    energies = data_file['all_energies'][NUM_GOOD_TRAIN_EVENTS_3:NUM_GOOD_TRAIN_EVENTS_3 + NUM_GOOD_TEST_EVENTS_3]
    return hist_fill(np.log10(energies), split_false=split)

def output_distribution(split):
    return hist_fill(predictions[np.where(labels == 1)], k40=True, split_false=split)

def histogram(distribution, bins, split=True, xlabel = '', normed=True, domain=None): 
    if not xlabel: xlabel = distribution.__name__.split('_')[0]
    dist = distribution(split=split)
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
        plot_list.append((ax1, efm, 'electron as muon'))
        plot_list.append((ax1, efk, 'electron as k40'))
        plot_list.append((ax2, mfe, 'muon as electron'))    
        plot_list.append((ax2, mfk, 'muon as k40'))    
    plot_list.append((ax1, eg, 'electron correct'))
    plot_list.append((ax2, mg, 'muon correct'))    


    for ax, data, label in reversed(plot_list):    
        h, b = np.histogram(data, bins=bins, range=domain, density=normed)
        #ax.hist(data, bins=bins, range=domain, density=normed, label=label, histtype='step')
        print 1/np.sqrt(h)
        ax.errorbar(x=b[:-1], y=h, yerr=1/np.sqrt(h), range=domain, label=label)
        ax.set_title(distribution.__name__ + ' ' + label.split()[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number events')
        ax.legend()
    plt.show()
        
def plot_acc_cost():
    path = PATH + "data/results/%s/" % title
   
    test_every = 10

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Cost/Loss as fuction of training epochs')
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Cross entropy per event')
    try:
        data = make_list_from_txt(PATH + 'data/results/%s/cost_train.txt' % title)
        epochs = len(data)
        ax1.plot(data, label='train')
    except IOError: 
        pass
    try:
        data = make_list_from_txt(PATH + 'data/results/%s/cost_test.txt' % title)
        ranger = range(epochs - 10 * len(data), epochs, test_every) 
        ax1.plot(ranger, data, label='test')
    except IOError: 
        pass
    ax1.legend()

    try: 
        data = make_list_from_txt(PATH + 'data/results/%s/acc_train.txt' % title)
        ranger = range(epochs - len(data), epochs) 
        ax2.plot(ranger, data, label='train')
    except IOError:
        pass 
    try: 
        data = make_list_from_txt(PATH + 'data/results/%s/acc_test.txt' % title)
        ranger = range(epochs - 10 * len(data), epochs, test_every) 
        ax2.plot(ranger, data, label='test')
    except IOError:
        pass 
    ax2.set_xlim([0, epochs])
    ax2.set_ylim([0, 1])
    ax2.set_title('Accuracy on training sets')
    ax2.legend()
    ax2.set_xlabel('Number of epochs epochs')
    ax2.set_ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
#    plot_acc_cost()
    plot_confusion_matrix()
#    histogram(output_distribution, bins=40, domain=(0,1))
#    histogram(energie_distribution, bins=50, xlabel='$\log(E)$', normed=False)
#    histogram(nhits_distribution, bins=50, xlabel='$\log(n)$')
#    
#    histogram(theta_distribution, bins=50, xlabel=r'$\cos(\theta)$')
#    histogram(phi_distribution, bins=50, xlabel='$\phi$')
#    positions()
    pass 
