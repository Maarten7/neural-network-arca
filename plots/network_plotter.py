import h5py 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools
from helper_functions import *

matplotlib.rcParams.update({
    'font.size': 16, 
    'pgf.rcfonts': False,
    'font.family': 'serif',
    'figure.figsize': [8, 6],
    'figure.autolayout': True,
    })

dens = False 
save = False 
"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"

# hdf5 files met (meta)data
pred_file_atm = h5py.File(PATH + 'results/temporal_atm/all_400ns_with_ATM_test_result_8.hdf5', 'r')

data_file = h5py.File(PATH + 'data/hdf5_files/all_400ns_with_ATM_test.hdf5', 'r')
#data_file = h5py.File(PATH + 'data/hdf5_files/all_400ns_with_ATM_validation.hdf5', 'r')

# Network output
predictions_atm   = pred_file_atm['all_test_predictions']

# alle informatie van alle events
labels   = data_file['all_labels'].value
energies = data_file['all_energies'].value
num_hits = data_file['all_num_hits'].value
types    = data_file['all_types'].value
triggers = data_file['all_masks'].value
num_muons= data_file['all_num_muons'].value
muons_th = data_file['all_muons_th'].value
var      = data_file['all_vars'].value
weights  = data_file['all_weights'].value
#
directions = data_file['all_directions'].value
theta = np.arctan2(directions[:,2],np.sqrt(np.sum(directions[:,0:2]**2, axis=1))) - np.pi / 2.
phi = np.arctan2(directions[:,1], directions[:,0])

# Predictions in to classes
l_true = np.argmax(labels, axis=1)
l_pred_atm = np.argmax(predictions_atm, axis=1)
eq     = l_true==l_pred_atm


def plot_normelized_with_error(bins, tot_dis, par_dis, ax, label):
    error =  par_dis / tot_dis.astype(float) * np.sqrt( 1./ par_dis - 1./tot_dis)
    ax.plot(bins[:-1], par_dis / tot_dis.astype(float), label=label, drawstyle='steps') 
    ax.fill_between(bins[:-1], par_dis / tot_dis.astype(float) - error, par_dis / tot_dis.astype(float) + error, alpha=0.1, step='pre') 
    return 0

def plot_confusion_matrix(pred):
    cm = confusion_matrix(l_true, pred)
    print cm
    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ, summ))
    cm = (cm / summ) * 100
    err_cm = cm * np.sqrt( 1. / cm + 1. / summ) 

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(4)

    classes = ['Shower', 'Track', 'K40', 'Atm Mu']
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.1f} $\pm$ {1:.1f} %".format(cm[i,j], err_cm[i,j]), horizontalalignment='center', color='red')

    plt.show()
    return 0

def histogram_classified_as(data_histogram, xlabel, split=True):
    fig, (ax1, ax2) = plt.subplots(1,2)
    #### shower
    he, be  = np.histogram(data_histogram[np.where( (l_true == 0)                 )], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred_atm == 0) )], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred_atm == 1) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred_atm == 2) )], bins=be, range=(be.min(), be.max()), density=dens)
    hea, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred_atm == 3) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax1, label='as shower')
    plot_normelized_with_error(be, he, hem, ax1, label='as track ')
    plot_normelized_with_error(be, he, hea, ax1, label='as atm muon')
    plot_normelized_with_error(be, he, hek, ax1, label='as K40')


    ax1.set_ylim(0,1)
    ax1.legend()
    ax1.set_title('Classification shower')
    ax1.set_ylabel('Fraction of shower events classified as')
    ax1.set_xlabel(xlabel)
    #### track
    he, be  = np.histogram(data_histogram[np.where( (l_true == 1)                 & split)], bins=30, density=dens)
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred_atm == 0) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred_atm == 1) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred_atm == 2) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hea, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred_atm == 3) & split)], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax2, label='as shower')
    plot_normelized_with_error(be, he, hem, ax2, label='as track')
    plot_normelized_with_error(be, he, hea, ax2, label='as atm muon')
    plot_normelized_with_error(be, he, hek, ax2, label='as K40')

    ax2.set_ylim(0,1)
    ax2.legend()
    ax2.set_title('Classification track')
    ax2.set_ylabel('Fraction of track events classified as')
    ax2.set_xlabel(xlabel)
    fig.set_size_inches(11.69, 8.27, forward=False)
    if save:
        fig.savefig('class_as' + xlabel + '.pdf')
    plt.show()

def histogram_muon_as(data_histogram, xlabel, split=True):
    fig, ax1 = plt.subplots(1,1)
    he, be  = np.histogram(data_histogram[np.where(                     (l_true == 3) )], bins=60, density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_pred_atm == 3) & (l_true == 3) )], bins=be, range=(be.min(), be.max()), density=dens)
    hen, _  = np.histogram(data_histogram[np.where( (l_pred_atm == 0) & (l_true == 3) )], bins=be, range=(be.min(), be.max()), density=dens)
    het, _  = np.histogram(data_histogram[np.where( (l_pred_atm == 1) & (l_true == 3) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_pred_atm == 2) & (l_true == 3) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hem, ax1, label='atm')
    plot_normelized_with_error(be, he, hen, ax1, label='show')
    plot_normelized_with_error(be, he, het, ax1, label='trac')
    plot_normelized_with_error(be, he, hek, ax1, label='k40')

    ax1.set_ylim(-0.01,1.01)
    ax1.legend()
    ax1.set_title('Classification atm muon')
    ax1.set_ylabel('Fraction of atm events triggered as')
    ax1.set_xlabel(xlabel)
    if save:
        fig.savefig('atm_as' + xlabel + '.pdf')
    plt.show()

    return 0
    
# all triggered events
def histogram_trigger(data_histogram, xlabel):
    fig, ax1 = plt.subplots(1,1)

    nu = ((l_true==0) | (l_true==1))
    atm = l_true==3
    ktr = ((l_pred_atm==0) | (l_pred_atm==1))
    jtr = triggers!=0

    set = atm
    he, be  = np.histogram(data_histogram[set], bins=60, density=dens)
    hem, _  = np.histogram(data_histogram[set & ktr], bins=be, range=(be.min(), be.max()), density=dens)
    hen, _  = np.histogram(data_histogram[set & jtr ], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hem, ax1, label='KM3NNeT')
    plot_normelized_with_error(be, he, hen, ax1, label='JTrigger')

    ax1.set_ylim(-0.01,1.01)
    #ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title('Triggering of atmospheric muon bundles')
    ax1.set_ylabel('Fraction of events triggered')
    ax1.set_xlabel(xlabel)
    if save:
        fig.savefig('trigger_' + xlabel + '.pdf')
    plt.show()

def trigger_rate(pred):
    cm = confusion_matrix(l_true, pred)
    tblocks_K40 = cm[2].sum()
    time_k40 = tblocks_K40 * 2e-5
    time_atm = 22670. # sec
    tblocks_triggerd_k40 = cm[2,0:2].sum()
    tblocks_triggerd_atm = cm[3,0:2].sum() 
    print 'KM3NNeT k40\t %f Hz' % (tblocks_triggerd_k40 / time_k40)
    print 'KM3NNeT atm\t %f Hz' % (tblocks_triggerd_atm / time_atm)
    tblocks_triggerd = ((triggers != 0) & (l_true == 3)).sum()
    print 'JTrigger k40\t %f Hz' % (0)
    print 'JTrigger atm\t %f Hz' % (tblocks_triggerd / time_atm)

atm = np.where(l_true==3)
trigger_rate(l_pred_atm)


def weighted_flux():
    nu = ((l_true==0) | (l_true==1))
    data_histogram = np.log10(energies)
    he, bins  = np.histogram(data_histogram[nu], bins=60, density=dens, )
    plt.semilogy(bins[:-1], he, drawstyle='steps', label='normal') 
    he, bins =np.histogram(data_histogram[nu], weights= 1. / weights[nu], bins=60, density=dens, )
    plt.semilogy(bins[:-1], he, drawstyle='steps', label='weighted') 
    plt.legend()
    #plt.yscale('log')
    plt.show()

#plot_confusion_matrix(l_pred_atm)

#histogram_classified_as(np.log10(num_hits), 'log N hits')
#histogram_muon_as(np.log10(num_hits), 'log N hits')

#histogram_muon_as(muons_th, 'N muons > threshold hits')
#histogram_classified_as(phi, 'phi')
#histogram_muon_as(phi, 'phi')

#histogram_classified_as(theta, 'theta')
#histogram_classified_as(np.cos(theta), 'cos theta')
#histogram_muon_as(np.cos(theta), 'cos theta')
#
histogram_trigger(np.log10(num_hits), 'log N hits',)
#histogram_trigger(np.log10(energies), 'log N hits',)
#histogram_trigger(np.cos(theta), 'cos theta')
#histogram_trigger(muons_th, 'N muons > threshold hits')

fout = np.where((l_true==3)&(l_pred_atm==2))
goed = np.where((l_true==3)&(l_pred_atm==3))

def plot_sum_tot(i):
    tots = data_file['all_tots'][i]
    bins = data_file['all_bins'][i]
    print 'nh', num_hits[i]
    print 'nm', num_muons[i]
    print 'th', theta[i]
    bins = np.vstack(bins)

    sum_tot = [tots[np.where(bins[0]==i)].sum() for i in range(50)]
    
    c = list() 
    for j in range(50):
        a = tots[np.where(bins[0]==j)] 
        b = [np.sqrt(np.sum(np.square(a[i:i+3]))) for i in range(0,len(a),3)]
        c.append(b[0])

    plt.plot(c)
    #plt.ylim(-100,10)
    plt.show()


print 'total acc', eq.sum() / float(eq.size)
