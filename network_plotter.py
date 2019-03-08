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

pred_file_atm = h5py.File(PATH + 'results/temporal_atm/all_400ns_with_ATM_test_result_8.hdf5', 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/test_file.hdf5', 'r')


predictions_atm   = pred_file_atm['all_test_predictions']
# alle informatie van alle events
labels   = data_file['all_labels'].value
energies = data_file['all_energies'].value
num_hits = data_file['all_num_hits'].value
triggers = data_file['all_masks'].value
num_muons= data_file['all_num_muons'].value
muons_th = data_file['all_muon_th'].value
weights  = data_file['all_weights'].value
types = data_file['all_types'].value
#
directions = data_file['all_directions'].value
theta = np.arctan2(directions[:, 2], np.sqrt(np.sum(directions[:, 0:2]**2, axis=1))) - np.pi / 2.
phi   = np.arctan2(directions[:, 1], directions[:, 0])

# Predictions in to classes
l_true = np.argmax(labels, axis=1)
l_pred = np.argmax(predictions_atm, axis=1)
eq     = l_true==l_pred


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

def histo(data, set, log, lbl, lp, axi):
    he, be  = np.histogram(data[set], bins=50, density=dens)
    b = 10 ** be if log else be

    hee, _  = np.histogram(data[set & (l_pred == lp)], bins=be, range=(be.min(), be.max()), density=dens)
    plot_normelized_with_error(b, he, hee, axi, label=lbl)

def histogram_classified_as(data_histogram, xlabel, log, atm):
    #### shower
    if not atm:
        fig, (ax1, ax2) = plt.subplots(1,2)
        plot_list = [(0, "showers", ax1), (1, "tracks", ax2),]
    else: 
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        plot_list = [(0, "showers", ax1), (1, "tracks", ax2),]
        plot_list.append((3, "atmospheric muons", ax3))

    for typ, string, axi in plot_list:
        set = l_true==typ

        for lbl, lp in [('shower', 0), ('track', 1), ('atm. muon', 3), ('K40', 2)]:
            histo(data=data_histogram, set=set, log=log, lbl=lbl, lp=lp, axi=axi)

        axi.set_ylim(-0.01,1.01)
        axi.legend()
        if log: axi.set_xscale('log')
        axi.set_title('Classification %s' % string)
        axi.set_ylabel('Fraction of %s events classified as' % string)
        axi.set_xlabel(xlabel)

    if save:
        fig.savefig('class_as' + xlabel + '.pdf')
    plt.show()

def k40():
    fig, axi = plt.subplots(1,1)
    set = l_true == 2
    data_histogram=num_hits
    for lbl, lp in [('shower', 0), ('track', 1), ('atm. muon', 3), ('K40', 2)]:
        histo(data=data_histogram, set=set, log=False, lbl=lbl, lp=lp, axi=axi)

    axi.set_ylim(-0.01,1.01)
    axi.legend()
    axi.set_title('Classification %s' % "K40")
    axi.set_ylabel('Fraction of %s events classified as' % "K40")
    plt.show()

# all triggered events
def histogram_trigger(data_histogram, xlabel, set, log):
    ktr = ((l_pred==0) | (l_pred==1))
    jtr = triggers!=0

    fig, ax1 = plt.subplots(1,1)
    he, be  = np.histogram(data_histogram[set], bins=60, density=dens)
    hem, _  = np.histogram(data_histogram[set & ktr], bins=be, range=(be.min(), be.max()), density=dens)
    hen, _  = np.histogram(data_histogram[set & jtr], bins=be, range=(be.min(), be.max()), density=dens)

    b = 10 ** be if log else be
    plot_normelized_with_error(b, he, hem, ax1, label='KM3NNeT')
    plot_normelized_with_error(b, he, hen, ax1, label='JTrigger')

    ax1.set_ylim(-0.01,1.01)
    if log: ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title('Triggering')
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

def weighted_flux():
    nu  = ((l_true==0) | (l_true==1))
    ktr = ((l_pred==0) | (l_pred==1))
    jtr = triggers!=0
    data_histogram = np.log10(energies)
#    he, bins  = np.histogram(data_histogram[nu], bins=60, density=dens, )
#    plt.semilogy(bins[:-1], he, drawstyle='steps', label='normal') 
    he, be =np.histogram(data_histogram[nu], weights=weights[nu], bins=60, density=dens, )
    b = 10 ** be[:-1]
    plt.plot(b, he, drawstyle='steps', label='Total') 

    he, bins =np.histogram(data_histogram[nu & ktr], weights=weights[nu & ktr], bins=be, density=dens, )
    plt.plot(b, he, drawstyle='steps', label='KM3NNeT') 

    he, bins =np.histogram(data_histogram[nu & jtr], weights=weights[nu & jtr], bins=be, density=dens, )
    plt.plot(b, he, drawstyle='steps', label='Default trigger') 

    plt.yscale('log')
    plt.xscale('log')
    plt.title('Triggering')
    plt.ylabel('Flux')
    plt.xlabel('E')
    plt.legend()
    plt.show()

nu  = ((l_true==0) | (l_true==1))
atm = l_true==3

trigger_rate(l_pred)

#weighted_flux()
#plot_confusion_matrix(l_pred)

#histogram_classified_as(np.log10(num_hits), '#Monte Carlo hits', log=True, atm=True)
#histogram_classified_as(np.log10(energies), r'$E_{\nu}$', log=True, atm=True)
#histogram_classified_as(np.cos(theta), r'$\cos(\theta_{\nu})$', log=False, atm=True)
#histogram_classified_as(muons_th, 'm', log=False, atm=True)
#
#histogram_trigger(np.log10(num_hits), '#Monte Carlo hits', set=nu, log=True)
#histogram_trigger(np.log10(energies), r'$E_{\nu}$', set=nu, log=True)
#histogram_trigger(np.cos(theta), r'$\cos(\theta_{\nu})$', set=nu, log=False)
#
#histogram_trigger(np.log10(num_hits), '#Monte Carlo hits', set=atm, log=True)
#histogram_trigger(np.cos(theta), r'$\cos(\theta_{\nu})$', set=atm, log=False)
#histogram_trigger(muons_th, 'N muons > threshold hits', set=atm, log=False )
