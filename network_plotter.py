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
pred_file = h5py.File(PATH + 'results/temporal/20000ns_400ns_test_result_temporal.hdf5', 'r')
pred_file2 = h5py.File(PATH + 'results/temporal/20000ns_250ns_test_result_temporal.hdf5', 'r')
data_file = h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5', 'r')


NUM_TEST_EVENTS -= 23410
# Network output
predictions = pred_file['all_test_predictions'][:NUM_TEST_EVENTS]
predictions2 = pred_file2['all_test_predictions'][:NUM_TEST_EVENTS]

# alle informatie van alle events
labels   = data_file['all_labels'][:NUM_TEST_EVENTS]
energies = data_file['all_energies'][:NUM_TEST_EVENTS]
num_hits = data_file['all_num_hits'][:NUM_TEST_EVENTS]
types    = data_file['all_types'][:NUM_TEST_EVENTS]
triggers = data_file['all_masks'].value

positions  = data_file['all_positions'][:NUM_TEST_EVENTS]
directions = data_file['all_directions'][:NUM_TEST_EVENTS]


# ruimtelijke informatie van neutrino
afstand = np.sqrt(np.sum(positions ** 2, axis=1))
theta   = np.arctan2(directions[:,2],np.sqrt(np.sum(directions[:,0:2]**2, axis=1)))
phi     = np.arctan2(directions[:,1], directions[:,0]) 
inward  = np.sum(positions * directions, axis=1) < 0
outward = np.sum(positions * directions, axis=1) > 0
Rxy     = np.sqrt(np.sum(positions[:,0:2] ** 2, axis=1))

# Predictions in to classes
l_true = np.argmax(labels, axis=1)
l_pred = np.argmax(predictions, axis=1)
l_pred2 = np.argmax(predictions2, axis=1)
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
    summ = np.column_stack((summ,summ,summ))
    cm = (cm / summ) * 100
    err_cm = cm * np.sqrt( 1. / cm + 1. / summ) 

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    #plt.colorbar()
    tick_marks = np.arange(3)

    classes = ['Shower', 'Track', 'K40']
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
    hee, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 0) )], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 1) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred == 2) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax1, label='as shower 400')
    plot_normelized_with_error(be, he, hem, ax1, label='as track 400')
    #plot_normelized_with_error(be, he, hek, ax1, label='as K40')

    hee, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred2 == 0) )], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred2 == 1) )], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 0) & (l_pred2 == 2) )], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax1, label='as shower 250')
    plot_normelized_with_error(be, he, hem, ax1, label='as track 250')
    #plot_normelized_with_error(be, he, hek, ax1, label='as K40')

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

    plot_normelized_with_error(be, he, hee, ax2, label='as shower 400')
    plot_normelized_with_error(be, he, hem, ax2, label='as track 400')
    #plot_normelized_with_error(be, he, hek, ax2, label='as K40')

    hee, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred2 == 0) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hem, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred2 == 1) & split)], bins=be, range=(be.min(), be.max()), density=dens)
    hek, _  = np.histogram(data_histogram[np.where( (l_true == 1) & (l_pred2 == 2) & split)], bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(be, he, hee, ax2, label='as shower 250')
    plot_normelized_with_error(be, he, hem, ax2, label='as track 250')
    #plot_normelized_with_error(be, he, hek, ax2, label='as K40')

    ax2.set_ylim(0,1)
    ax2.legend()
    ax2.set_title('Classification track')
    ax2.set_ylabel('Fraction of track events classified as')
    ax2.set_xlabel(xlabel)
    fig.set_size_inches(11.69, 8.27, forward=False)
    #fig.savefig('histogram_as_' + xlabel + '.pdf', dpi=500)
    plt.show()
    return 0
    
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
    #fig.savefig('histogram_split_types_' + xlabel + '.pdf', dpi=500)
    plt.show()
    return 0


# all triggered events
def histogram_trigger(data_histogram, xlabel):
    fig, ax1 = plt.subplots(1,1)
    he, be  = np.histogram(np.log10(data_histogram[np.where(                  (l_true != 2) )]), bins=60, density=dens)
    hen, _  = np.histogram(np.log10(data_histogram[np.where( (l_pred != 2)  & (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)
    he2, _  = np.histogram(np.log10(data_histogram[np.where( (l_pred2 != 2)  & (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)
    het, _  = np.histogram(np.log10(data_histogram[np.where( (triggers != 0)& (l_true != 2) )]), bins=be, range=(be.min(), be.max()), density=dens)

    plot_normelized_with_error(10 ** be, he, hen, ax1, label='KM3NNeT 400')
    plot_normelized_with_error(10 ** be, he, he2, ax1, label='KM3NNeT 250')
    plot_normelized_with_error(10 ** be, he, het, ax1, label='JTrigger')

    ax1.set_ylim(-0.01,1.01)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title('Trigger Efficientcy')
    ax1.set_ylabel('Fraction of events triggered')
    ax1.set_xlabel(xlabel)
    if save:
        fig.savefig('trigger_' + xlabel + '.pdf')
    plt.show()

    return he, hen, het, be

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

def animate_event(event_full):
    """Shows 3D plot of evt"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    event_full = np.sqrt(np.sum(np.square(event_full), axis=4))
    ims = []
    for i, event in enumerate(event_full):
        x, y, z = event.nonzero()
        k = event[event.nonzero()]
        sc = ax.scatter(x, y, z, zdir='z', c=k, cmap=plt.get_cmap('Blues'), norm=matplotlib.colors.LogNorm(0.1, 350))
        ims.append([sc])
    ax.set_xlim([0,13])
    ax.set_ylim([0,13])
    ax.set_zlim([0,18])
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    ax.set_zlabel('z index')
    plt.title('TTOT on DOM')
    fig.colorbar(sc)
    ani = animation.ArtistAnimation(fig, ims)
    plt.show()

def trigger_rate(pred):
    cm = confusion_matrix(l_true, pred)
    tblocks_K40 = cm[2].sum()
    time = tblocks_K40 * 2e-5
    tblocks_triggerd = cm[2,0:2].sum()
    print '%f Hz' % (tblocks_triggerd / time)

#trigger_rate(l_pred)
#trigger_rate(l_pred2)

#histogram_classified_as(np.log10(num_hits), 'log E', Rxy < 500)
#histogram_classified_as(np.log10(num_hits), 'log E', Rxy > 500)
#histogram_classified_as(np.log10(num_hits), 'log E', inward)
#histogram_classified_as(np.log10(num_hits), 'log E', outward)

#histogram_classified_as(afstand, 'R meters')
#histogram_classified_as(np.log10(afstand), 'log R ')
#histogram_classified_as(np.cos(theta), 'cos theta')
#histogram_classified_as(theta, 'theta')
#histogram_classified_as(phi, 'phi')

#histogram_classified_as(np.log10(energies), 'log E')
histogram_classified_as(np.log10(num_hits), 'log N hits')

#histogram_split_types(np.log10(energies), 'log E')
#histogram_split_types(np.log10(num_hits), 'log N hits')

#histogram_trigger(energies, r'$E_{\nu}$')
#histogram_trigger(num_hits, '# MC Hits')

#plot_confusion_matrix(l_pred)
#plot_confusion_matrix(trigger_conf_matrix())
