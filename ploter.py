import h5py 
import numpy as np
import importlib
import sys
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper_functions import *

#model = sys.argv[1]
#model = importlib.import_module(model)
#title = model.title

title = 'sum_tot'

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


def result_plot():
    z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
    predictions = z['predictions_bg']
    labels = z['labels_bg']
    events = z['events_bg']

    nhits = []
    q = h5py.File('mc_hits_test.hdf5', 'r')
    for root_file, _ in root_files(train=False, test=True):
        nhits.extend(q[root_file + 'nhits'].value)

    eg, ef, mg, mf  = [], [], [], []
    for i in range(len(predictions)):
        numhits = nhits[i] 

        ll = np.argmax(labels[i])
        lt = np.argmax(predictions[i])

        if ll == lt:
            if ll == 0:
                eg.append(numhits)
            else:
                mg.append(numhits)
        else:
            if ll == 0:
                ef.append(numhits)
            else:
                mf.append(numhits)
     
    return eg, ef, mg, mf

def output_distribution():
    z = h5py.File(PATH + 'data/results/%s/test_result_%s.hdf5' % (title, title), 'r')
    predictions = z['predictions_bg']
    labels = z['labels_bg']
    events = z['events_bg']

    eg, ef, mg, mf  = [], [], [], []
    for i in range(len(predictions)):
        ll = np.argmax(labels[i])
        lt = np.argmax(predictions[i])
        
        if ll == lt:
            #restult good
            if ll == 0:
                # electron
                eg.append(predictions[i][0])
            else:
                # muon
                mg.append(predictions[i][1])
        else:
            if ll == 0:
                ef.append(predictions[i][0])
            else:
                mf.append(predictions[i][1])
    
    return eg, ef, mg, mf
    


def histogram_plot():
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

if __name__ == '__main__':
#    plot_acc_cost()
#    histogram_plot()
    eg, ef, mg, mf = output_distribution()

    plt.hist(eg, bins=20, label='enu correct')
    plt.hist(ef, bins=20, label='enu false')
    plt.title('distribution output network electon neutrino')
    plt.legend()
    plt.show()

    plt.hist(mg, bins=20, label='munu correct')
    plt.hist(mf, bins=20, label='munu false')
    plt.title('distribution output network muon neutrino')
    plt.legend()
    plt.show()

