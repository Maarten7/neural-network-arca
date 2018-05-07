import h5py 
import numpy as np
import importlib
import sys
import importlib
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from helper_functions import *

#model = sys.argv[1]
#model = importlib.import_module(model)
#title = model.title

title = 'sum_tot'
title = 'three_classes_sum_tot'

#z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
#q = h5py.File(PATH + 'data/hdf5_files/bg_file_%s.hdf5' % title  )
#predictions = z['predictions_bg']
#labels = z['labels_bg']
#events = z['events_bg']

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

def hist_fill(list_value):
    eg, ef, mg, mf  = [], [], [], []
    ll = np.argmax(labels.value, axis=1)
    lt = np.argmax(predictions.value, axis=1)
    eq = np.equal(ll, lt)
    for i in range(len(predictions)):
        x = list_value[i] 
        typ = ll[i]
        pr = predictions[i][typ]
        if eq[i]: # correct prediciont
            if typ == 0 and 0.97 < pr < 1: # electon
                eg.append(x) 
            elif typ == 1 and 0.97 < pr < 1: # muon
                mg.append(x)
        if not eq[i]: # incorrect prediction
            if typ == 0 and 0 < pr < .03: # electron
                ef.append(x) 
            elif typ == 1 and 0 < pr < .03: # electron 
                mf.append(x)

    return eg, ef, mg, mf

def nhits_distribution():
    nhits = []
    for root_file, _ in root_files(train=False, test=True):
        nhits.extend(q[root_file + 'n_hits'].value)
    return hist_fill(nhits)

def energie_distribution():
    energies = []
    for root_file, _ in root_files(train=False, test=True):
        energies.extend(np.log(q[root_file + 'E'].value))

    return hist_fill(energies)

def pos_distribution(i):
    positions = []
    for root_file, _ in root_files(train=False, test=True):
        for pos in q[root_file + 'positions'].value:
            positions.append(pos[i])
    return hist_fill(positions)

def dir_distribution(i):
    directions= []
    for root_file, _ in root_files(train=False, test=True):
        for dir in q[root_file + 'directions'].value:
            directions.append(dir[i])
    return hist_fill(directions)

def theta_distribution():
    thetas = []
    for root_file, _ in root_files(train=False, test=True):
        for dir in q[root_file + 'directions'].value:
            dx, dy, dz = dir
            theta = np.arctan2(dz, math.sqrt(dx**2 + dy**2)) 
            thetas.append(np.cos(theta))
    return hist_fill(thetas) 

def phi_distribution():
    phis = []
    for root_file, _ in root_files(train=False, test=True):
        for dir in q[root_file + 'directions'].value:
            dx, dy, dz = dir
            phi = np.arctan2(dy, dx) 
            phis.append(phi) 
    return hist_fill(phis) 

def output_distribution():
    return hist_fill(predictions.value[np.where(labels.value == 1)])

def histogram(distribution, bins, xlabel, domain=None, i=0): 
    eg, ef, mg, mf = distribution()
    fig, (ax1, ax2) = plt.subplots(1,2)
    for ax, type_good, type_false, stype in [(ax1, eg, ef, 'electron'), (ax2, mg, mf, 'muon')]:    
        ax.hist(type_good, bins=bins, range=domain, normed=True, label='%s correct' % stype)
        ax.hist(type_false, bins=bins, range=domain, normed=True, label='%s false' % stype)
        ax.set_title(distribution.__name__ + ' ' + stype)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('number events')
        ax.legend()
    plt.show()
        
def plot_acc_cost():
    fig, (ax1, ax2) = plt.subplots(1,2)
    try:
        data = make_list_from_txt(PATH + 'data/results/%s/cost.txt' % title)
        ax1.plot(data)
    except IOError: 
        pass
    ax1.set_title('Cost/Loss as fuction of training epochs')
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Cross entropy')
    try:
        ax2.plot(make_list_from_txt(PATH + 'data/results/%s/acc.txt' % title))
    except IOError:
        pass 
    ax2.set_title('Accuracy on training sets')
    ax2.legend()
    ax2.set_xlabel('epochs')
    plt.show()

def histogram_n_hits():
    save_path = PATH + 'data/results/%s/' % title
    label=['e correct', 'e false', 'm correct', 'm false']
    eg, ef, mg, mf = result_plot()
    domain = (0, 500)

    h, b, _ = plt.hist([eg, ef], bins=50, range=domain, label=label[0:2], histtype='bar')
    h1, h2 = h
    acc = h1/(h1 + h2)
    
    plt.title('Number of hit distribution eCC and eNC')
    plt.ylabel('number of events')
    plt.xlabel('number of hits')
    plt.legend()
    plt.savefig(save_path + 'hist_e')
    plt.show()
    
    plt.plot(b[1:], acc)
    plt.title('eCC and eNC accuracy')
    plt.xlabel('number of hits')
    plt.ylabel('accuracy')
    plt.savefig(save_path + 'acc_e')
    plt.show()

    h, b, _ = plt.hist([mg, mf], bins=50, range=domain, label=label[2:4], histtype='bar')
    h1, h2 = h
    acc = h1/(h1 + h2)
    plt.title('Number of hit distribution muCC')
    plt.ylabel('number of events')
    plt.xlabel('number of hits')
    plt.legend()
    plt.savefig(save_path + 'hist_mu')
    plt.show()
    
    plt.plot(b[1:], acc)
    plt.title('muCC accuracy')
    plt.xlabel('number of hits')
    plt.ylabel('accuracy')
    plt.savefig(save_path + 'acc_mu')
    plt.show()
    
    plt.hist([eg, ef, mg, mf], bins=50, range=domain, label=label, histtype='barstacked')
    plt.title('Number of hit distribution (stacked)')
    plt.legend()
    plt.xlabel('number of hits')
    plt.ylabel('number of events')
    plt.savefig(save_path + 'hist_all_stacked')
    plt.show()
    
    h, b, _ = plt.hist([eg + mg, ef + mf], bins=50, range=domain, label=label, histtype='bar')
    h1, h2 = h
    acc = h1/(h1 + h2)
    plt.close()

    plt.title('Accuracy')
    plt.xlabel('number of hits')
    plt.ylabel('accuracy')
    plt.plot(b[1:], acc)
    plt.savefig(save_path + 'Acc_all')
    plt.show()

def positions():
    ll = np.argmax(labels.value, axis=1)
    lt = np.argmax(predictions.value, axis=1)
    eq = np.equal(ll, lt)
    
    positions = []
    for root_file, _ in root_files(train=False, test=True):
        for pos in q[root_file + 'positions'].value:
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
    plot_acc_cost()
#    histogram(output_distribution, bins=40, domain=(0,1), xlabel='output')
#    histogram(energie_distribution, bins=100,domain=None, xlabel='energie')
#    histogram(nhits_distribution, bins=100, domain=(0,200), xlabel='mc hits')
#    
#    histogram(theta_distribution, bins=50, domain=None, xlabel=r'$\cos(\theta)$')
#    histogram(phi_distribution, bins=50, domain=None, xlabel='$\phi$')
#    positions()

