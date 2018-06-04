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
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
model = sys.argv[1].replace('/', '.')[:-3]
try:
	model = importlib.import_module(model)
	title = getattr(model, "title") 
except ImportError:
	title = 'three_classes_sum_tot'

data_file = h5py.File(PATH_RANCE + 'data/hdf5_files/events_and_labels_%s.hdf5' % title)
pred_file = h5py.File(PATH_RANCE + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
meta_file = h5py.File(PATH_RANCE + 'data/hdf5_files/meta_data.hdf5')

predictions = pred_file['predictions']
labels = pred_file['labels']

ll = np.argmax(labels.value, axis=1)
lt = np.argmax(predictions.value, axis=1)
eq = np.equal(ll, lt)

def plot_confusion_matrix():
    cm = confusion_matrix(ll, lt)
    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ))
    cm = cm / summ

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(3)

    classes = ['shower', 'track', 'K40']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], '.3f'), horizontalalignment='center', color='red')

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
    nhits = []
    for root_file, _ in root_files(train=False, test=True):
        nhits.extend(meta_file[root_file + 'n_hits'].value)
    return hist_fill(nhits, split_false=split)

def energie_distribution(split):
    energies = []
    for root_file, _ in root_files(train=False, test=True):
        energies.extend(np.log(meta_file[root_file + 'E'].value))

    return hist_fill(energies, split_false=True)

def theta_distribution(split):
    thetas = []
    for root_file, _ in root_files(train=False, test=True):
        for dir in meta_file[root_file + 'directions'].value:
            dx, dy, dz = dir
            theta = np.arctan2(dz, math.sqrt(dx**2 + dy**2)) 
            thetas.append(np.cos(theta))
    return hist_fill(thetas, split_false=split) 

def phi_distribution(split):
    phis = []
    for root_file, _ in root_files(train=False, test=True):
        for dir in meta_file[root_file + 'directions'].value:
            dx, dy, dz = dir
            phi = np.arctan2(dy, dx) 
            phis.append(phi) 
    return hist_fill(phis, split_false=split) 

def output_distribution(split):
    return hist_fill(predictions.value[np.where(labels.value == 1)], k40=True, split_false=split)

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
	if host == 'rance':
		ax.hist(data, bins=bins, range=domain, normed=normed, label=label, histtype='step')
	else:
		ax.hist(data, bins=bins, range=domain, density=normed, label=label, histtype='step')
        ax.set_title(distribution.__name__ + ' ' + label.split()[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number events')
        ax.legend()
    plt.show()
        
def plot_acc_cost():
    path = PATH + "data/results/%s/" % title
    os.system("ls -l " + path + "_cost_train>" + path + "cost_train.txt")
    os.system("ls -l " + path + "_acc_train>" + path + "acc_train.txt")
    os.system("ls -l " + path + "_cost_test>" + path + "cost_test.txt")
    os.system("ls -l " + path + "_acc_test>" + path + "acc_test.txt")
   
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


def positions():
    positions = []
    for root_file, _ in root_files(train=False, test=True):
        for pos in meta_file[root_file + 'positions'].value:
            positions.append(pos)
    egx, efx, mgx, mfx = [], [], [], []
    egy, efy, mgy, mfy = [], [], [], []
    for i in range(len(predictions)):
        pr = predictions[i]
        x, y, z = positions[i]
        typ = ll[i]
        pr = predictions[i][typ]
        if eq[i]: # correct prediciont
            if typ == 0 and 0.97 < pr < 1: # electon
                egx.append(x)
                egy.append(y)
            elif typ == 1 and 0.97 < pr < 1: # muon
                mgx.append(x)
                mgy.append(y)
        if not eq[i]: # incorrect prediction
            if typ == 0 and 0 < pr < .03: # electron
                efx.append(x)
                efy.append(y)
            elif typ == 1 and 0 < pr < .03: # electron 
                mfx.append(x)
                mfy.append(y)
    plt.plot(egx, egy, 'g.')
    plt.plot(efx, efy, 'r.')
    plt.plot(mgx, mgy, 'g^')
    plt.plot(mfx, mfy, 'r^')
    plt.show()

if __name__ == '__main__':
#    plot_acc_cost()
#    histogram(output_distribution, bins=40, domain=(0,1))
#    histogram(energie_distribution, bins=100, xlabel='$\log(E)$')
#    histogram(nhits_distribution, bins=100, domain=(0,200))
##    
#    histogram(theta_distribution, bins=50, xlabel=r'$\cos(\theta)$')
#    histogram(phi_distribution, bins=50, xlabel='$\phi$')
    plot_confusion_matrix()
#    positions()
    pass 
