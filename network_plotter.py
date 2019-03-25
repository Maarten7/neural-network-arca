import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import itertools
from helper_functions import *

matplotlib.rcParams.update({
    'font.size': 16, 
<<<<<<< HEAD
    'font.family': 'serif',
    'pgf.rcfonts': True,
    'pgf.texsystem': "pdflatex",
    'figure.figsize': [10, 5], 
=======
    'pgf.rcfonts': False,
    'font.family': 'serif',
    'figure.figsize': [8, 6],
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9
    'figure.autolayout': True,
    })

dens = False 
save = True
"PLOTS energy and num_hits distribution of classified events. The energy and n hits distrubution is normalized"

<<<<<<< HEAD

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

=======
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

>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9
def plot_confusion_matrix(pred):
    fig, ax = plt.subplots(1)
    cm = confusion_matrix(l_true, pred)
<<<<<<< HEAD
    
    print cm 

    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ))
=======
    print cm
    summ = np.sum(cm, axis=1, dtype=float)
    summ = np.column_stack((summ,summ,summ, summ))
    cm = (cm / summ) * 100
    err_cm = cm * np.sqrt( 1. / cm + 1. / summ) 
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9

    err_cm = np.nan_to_num(100 * cm / summ * np.sqrt( 1./ cm - 1. / summ))
    cm = 100 * cm / summ
    

    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title('Normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(4)

    classes = ['Shower', 'Track', 'K40', 'Atm Mu']
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

<<<<<<< HEAD
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
=======
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
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9
    ax1.set_ylabel('Fraction of events triggered')
    ax1.set_xlabel('E [GeV]')
    ax2.set_xlabel('Monte Carlo hits')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    
    if save:
<<<<<<< HEAD
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
=======
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
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9

nu  = ((l_true==0) | (l_true==1))
atm = l_true==3

<<<<<<< HEAD
histogram_trigger()
=======
trigger_rate(l_pred)
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9

#weighted_flux()
#plot_confusion_matrix(l_pred)
<<<<<<< HEAD
#plot_confusion_matrix(trigger_conf_matrix())
=======

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
>>>>>>> de9ef3a51fefc3bd8bd040ddb78ce82f89ef83e9
