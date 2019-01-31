import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools
from helper_functions import *

matplotlib.rcParams.update({
    'font.size': 16, 
    'font.family': 'serif',
    'pgf.rcfonts': True,
    'pgf.texsystem': "pdflatex",
    'figure.figsize': [10, 5], 
    'figure.autolayout': True,
    })

dens = False 
save = True
"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"


# Network output
l_true = []
l_pred = []
energies = []
num_hits = []
triggers = []

with open('plotinfo.txt', 'r') as pfile:
    for line in pfile:
        l, p, t, e, n = line.split(',')
        l_true.append(int(l))
        l_pred.append(int(p))
        triggers.append(int(t))
        energies.append(float(e))
        num_hits.append(int(n))

positions = np.zeros((len(l_true), 3))
directions = np.zeros((len(l_true), 3))
with open('posdir.txt', 'r') as pfile:
    for i, line in enumerate(pfile):
        x, y, z, dx, dy, dz = line.split(',')
        positions[i] = np.array([float(x), float(y), float(z)])
        directions[i] = np.array([float(dx), float(dy), float(dz)])

l_true = np.array(l_true)
l_pred = np.array(l_pred)
triggers = np.array(triggers)
energies = np.array(energies)
num_hits = np.array(num_hits)
    
# ruimtelijke informatie van neutrino
afstand = np.sqrt(np.sum(positions ** 2, axis=1))
theta = np.arctan2(directions[:,2],np.sqrt(np.sum(directions[:,0:2]**2, axis=1)))
phi = np.arctan2(directions[:,1], directions[:,0]) 
inward = np.sum(positions * directions, axis=1) < 0
outward = np.sum(positions * directions, axis=1) > 0
Rxy = np.sqrt(np.sum(positions[:,0:2] ** 2, axis=1))
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]


def plot_normelized_with_error(bins, total, partial, ax, label):
    tot = total.astype(float)
    par = partial.astype(float)
    fin = par / tot 

    error =  par / tot * np.sqrt(1/ par - 1/tot)

    ax.plot(bins[:-1], fin, label=label, drawstyle='steps') 
    ax.fill_between(bins[:-1], fin - error, fin + error, alpha=0.3, step='pre') 
    return 0

def plot_confusion_matrix(pred):
    fig, ax = plt.subplots(1)
    cm = confusion_matrix(l_true, pred)
    
    print cm 

    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ))

    err_cm = np.nan_to_num(100 * cm / summ * np.sqrt( 1./ cm - 1. / summ))
    cm = 100 * cm / summ
    

    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title('Normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(3)

    classes = ['Shower', 'Track', 'K40']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.1f} $\pm$ {1:.1f} %".format(cm[i,j], err_cm[i,j]), horizontalalignment='center', color='red')

    if save:
        fig.savefig("confusionmatrix.pgf")
    plt.show()
    return 0

split1 = (Rxy < 500) & (z < 400) & (z > -400)
split2 = (Rxy > 500) & (z > 400) & (z < -400)
def detector_regions():
    fig, (ax1, ax2) = plt.subplots(1,2)
    data = energies
    bins = 60
    ax = ax1
    split1 = (Rxy < 500) & (z < 400) & (z > -400)
    split2 = (Rxy > 500) & (z > 400) & (z < -400)
    label1 = 'in box' 
    label2 = 'out box' 
    he, be  = np.histogram(np.log10(data[np.where( (l_true == 0))]), bins=bins, density=dens)

    hee, _  = np.histogram(np.log10(data[np.where( (l_true == 0) & (l_pred == 0) & split1)]), bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(10 ** be, he, hee, ax, label=label1)

    hee, _  = np.histogram(np.log10(data[np.where( (l_true == 0) & (l_pred == 0) & split2)]), bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(10 ** be, he, hee, ax, label=label2)

    ax.set_ylim(0,1)
    ax.legend()
    ax.set_title('Classification of shower events')
    ax.set_ylabel('Fraction events classified as')
    ax.set_xscale('log')

    ax = ax2
    he, be  = np.histogram(np.log10(data[np.where( (l_true == 1))]), bins=bins, density=dens)

    hee, _  = np.histogram(np.log10(data[np.where( (l_true == 1) & (l_pred == 1) & split1)]), bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(10 ** be, he, hee, ax, label=label1)

    hee, _  = np.histogram(np.log10(data[np.where( (l_true == 1) & (l_pred == 1) & split2)]), bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(10 ** be, he, hee, ax, label=label2)

    ax.set_ylim(0,1)
    ax.legend()
    ax.set_title('Classification of track events')
    ax.set_ylabel('Fraction events classified as')
    ax.set_xscale('log')

    plt.show()  


def histogram_classified_as(split=True):
    fig, (ax1, ax2) = plt.subplots(1,2)

    #### shower
    data_histogram = energies
    he, be  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0)                 )]), bins=30, density=dens)
    hee, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hee, ax1, label='Shower')
    plot_normelized_with_error(10 ** be, he, hem, ax1, label='Track')
    plot_normelized_with_error(10 ** be, he, hek, ax1, label='K40')
    ax1.set_ylim(0,1)
    ax1.legend()
    #ax1.set_title('Classification of shower events')
    ax1.set_ylabel('Fraction events classified as')
    ax1.set_xlabel('E [GeV]')
    ax1.set_xscale('log')

    #### shower
    data_histogram = num_hits 
    he, be  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0)                 )]), bins=30, density=dens)
    hee, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 0) )]), bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 1) )]), bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 0) & (l_pred == 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hee, ax2, label='Shower')
    plot_normelized_with_error(10 ** be, he, hem, ax2, label='Track')
    plot_normelized_with_error(10 ** be, he, hek, ax2, label='K40')
    ax2.set_ylim(0,1)
    ax2.legend()
    #ax2.set_title('Classification of shower events')
    #ax2.set_ylabel('Fraction events classified as')
    ax2.set_xlabel('Monte Carlo hits')
    ax2.set_xscale('log')

    ax1.set_ylim(-0.01,1.01)
    ax2.set_ylim(-0.01,1.01)
    ax2.set_yticklabels([])
    fig.suptitle('Classification of shower events', va='top')
    if save:
        fig.savefig('histogram_as_shower' + '.pgf')
    plt.show()

    fig, [ax3, ax4] = plt.subplots(1,2)
    #### track
    data_histogram = energies
    he, be  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1)                 & split)]), bins=30, density=dens)
    hee, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 0) & split)]), bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 1) & split)]), bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 2) & split)]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hee, ax3, label='Shower')
    plot_normelized_with_error(10 ** be, he, hem, ax3, label='Track')
    plot_normelized_with_error(10 ** be, he, hek, ax3, label='K40')
    ax3.set_ylim(0,1)
    ax3.set_xscale('log')
    ax3.legend()
    ax3.set_ylabel('Fraction events classified as')
    ax3.set_xlabel('E [GeV]')

    #### track
    data_histogram = num_hits
    he, be  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1)                 & split)]), bins=30, density=dens)
    hee, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 0) & split)]), bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 1) & split)]), bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(np.log10(data_histogram[np.where( (l_true == 1) & (l_pred == 2) & split)]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hee, ax4, label='Shower')
    plot_normelized_with_error(10 ** be, he, hem, ax4, label='Track')
    plot_normelized_with_error(10 ** be, he, hek, ax4, label='K40')
    ax4.set_ylim(0,1)
    ax4.set_xscale('log')
    ax4.legend()
    ax4.set_xlabel('Monte carlo hits')
    
    ax3.set_ylim(-0.01,1.01)
    ax4.set_ylim(-0.01,1.01)

    ax4.set_yticklabels([])
    fig.suptitle('Classification of track events', va='top')
    if save:
        fig.savefig('histogram_as_track' + '.pgf')
    plt.show()
    return 0
    
# all triggered events
def histogram_trigger():
    fig, (ax1, ax2) = plt.subplots(1,2)

    data_histogram = energies
    he, be  = np.histogram(np.log10(data_histogram[np.where(                  (l_true != 2) )]), bins=60, density=dens)
    hen, _  = np.histogram(np.log10(data_histogram[np.where( (l_pred != 2)  & (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)
    het, _  = np.histogram(np.log10(data_histogram[np.where( (triggers != 0)& (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hen, ax1, label='KM3NNeT')
    plot_normelized_with_error(10 ** be, he, het, ax1, label='JTrigger')

    data_histogram = num_hits 
    he, be  = np.histogram(np.log10(data_histogram[np.where(                  (l_true != 2) )]), bins=60, density=dens)
    hen, _  = np.histogram(np.log10(data_histogram[np.where( (l_pred != 2)  & (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)
    het, _  = np.histogram(np.log10(data_histogram[np.where( (triggers != 0)& (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hen, ax2, label='KM3NNeT')
    plot_normelized_with_error(10 ** be, he, het, ax2, label='JTrigger')

    ax1.set_ylim(-0.01,1.01)
    ax2.set_ylim(-0.01,1.01)
    ax2.set_yticklabels([])
    ax1.legend()
    fig.suptitle('Trigger Efficiency\n', va='top')
    ax1.set_ylabel('Fraction of events triggered')
    ax1.set_xlabel('E [GeV]')
    ax2.set_xlabel('Monte Carlo hits')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    
    if save:
        fig.savefig('trigger' + '.pgf')
    plt.show()
    return be, he, hen, het 

def trigger_conf_matrix():
    trigger_pred = np.zeros(NUM_TEST_EVENTS)
    for i in range(NUM_TEST_EVENTS):
        #K40
        if triggers[i] == 0:
            trigger_pred[i] = 2
        #SHOWER
        if triggers[i] == 2:
            trigger_pred[i] = 0
        #TRACK
        if triggers[i] == 16:
            trigger_pred[i] = 1
        # BOTH
        if triggers[i] == 18:
            # true = shower
            if l_true[i] == 0:
                trigger_pred[i] = 0
            # true = track
            if l_true[i] == 1:
                trigger_pred[i] = 1

            # true = k40
            if l_true[i] == 2 and l_pred[i] != 2:
                trigger_pred[i] = l_pred[i] 
            if l_true[i] == 2 and l_pred[i] == 2:
                trigger_pred[i] = 0 
    return trigger_pred

#histogram_classified_as(np.log10(energies), 'log E', Rxy < 250)
#histogram_classified_as(np.log10(energies), 'log E', ((250 < Rxy) & ( Rxy < 500)))
#histogram_classified_as(np.log10(energies), 'log E', Rxy > 500)
#histogram_classified_as(np.log10(energies), 'log E', outward)
#histogram_classified_as(np.log10(energies), 'log E', inward)

#histogram_classified_as(afstand, 'R meters')
#histogram_classified_as(np.log10(afstand), 'log R ')
#histogram_classified_as(np.cos(theta), 'cos theta')
#histogram_classified_as(theta, 'theta')
#histogram_classified_as(phi, 'phi')

histogram_classified_as()

#histogram_split_types(np.log10(energies), 'log E')
#histogram_split_types(np.log10(num_hits), 'log N hits')

histogram_trigger()

#plot_confusion_matrix(l_pred)
#plot_confusion_matrix(trigger_conf_matrix())
