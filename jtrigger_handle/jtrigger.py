# jtrigger.py
# Maarten Post
""" Takes the test data set and looks what events are triggered by JTriggerEfficenty
    in order to compare it with KM3NNET"""
from ROOT import *
import aa
import numpy as np
import sys
import h5py

import matplotlib.pyplot as plt

PATH = "/user/postm/neural-network-arca/"

EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40']
NUM_CLASSES = 3
NUM_MINI_TIMESLICES = 80

def pmt_id_to_dom_id(pmt_id):
    #channel_id = (pmt_id - 1) % 31
    dom_id     = (pmt_id - 1) / 31 + 1
    return dom_id

def make_file_str(evt_type, i, J='JEW'):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_{2}_km3_v4_{0}_{1}.evt.root'
    n = path.format(evt_type, i, J)
    return n

def root_files(range, J='JEW'):
    """ outputs strings of all root_files"""
    for i in range:
        for evt_type in EVT_TYPES:
            n = make_file_str(evt_type, i, J=J)
            yield n, evt_type

def doms_hit_pass_threshold(mc_hits, threshold, pass_k40):
    """ checks if there a at least <<threshold>> doms
        hit by monte carlo hits. retuns true or false"""
    if len(mc_hits) == 0: return pass_k40 

    dom_id_set = set()
    for hit in mc_hits:
        dom_id = pmt_id_to_dom_id(hit.pmt_id)
        dom_id_set.add(dom_id)
        if len(dom_id_set) >= threshold:
            return True
    return False

def root_converter():
    print "\nroot converter"
    EventFile.read_timeslices = True
    k = 0
    t = 0
    for root_file, typ in root_files(range(13, 16), 'JEW'):
        print '\t', root_file
        code = root_file.split('/')[-1].split('.')[0].split('_')[-1]
        with open('roots_%s_%s.txt' % (code, typ), 'w') as tf:
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            t += len(f)
            kk = 0
            for i, evt in enumerate(f):

                if "K40" in typ:
                    E = 0
                else:
                    #E = evt.mc_trks[0].E
                    E = len(evt.mc_hits)

                passes = doms_hit_pass_threshold(evt.mc_hits, 5, True)
                tf.write("%i, %i, %i\n" % (evt.frame_index, passes, E))
                if passes:
                    k += 1
                    kk += 1
            print "\t", len(f), kk
    print t, k

def trigger_converter():
    print "\ntrigger converter"
    EventFile.read_timeslices = False 
    total_trigger_events = 0

    for JTP, typ in root_files(range(13, 16), 'JTE'):
        if "K40" in JTP: continue
        print '\t', JTP
        code = JTP.split('/')[-1].split('.')[0].split('_')[-1]
        with open('triggers_%s_%s.txt' % (code, typ), 'w') as tf:
            f = EventFile(JTP)
            #f.use_tree_index_for_mc_reading = False
            print '\t', typ, code, len(f)
            total_trigger_events += len(f)
            for evt in f:
                tf.write("%i, %i, %i, %i\n" % (evt.frame_index, evt.trigger_counter, evt.trigger_mask, len(evt.mc_hits)))
    print 'total triggered', total_trigger_events

def trigger_gen(triggerfile):
    for line in triggerfile:
        frame_index, trigger_counter, mask, E = line.split(',')
        yield int(frame_index) - 1, int(trigger_counter) + 1, int(mask), int(E)

def root_gen(rootfile):
    for line in rootfile:
        line = line.split(',')
        frame_index, passed, E = line
        yield int(frame_index), int(passed), int(E)

def combine_trigger_and_root():
    print "\ncombine trigger and root"

    total_trigger = 0
    total_events = 0
    for j in [13, 14, 15]:
        for typ in EVT_TYPES:
            t = 0
            e = 0

            root_file = open("roots_%i_%s.txt" % (j, typ), 'r')
            trig_file = open("triggers_%i_%s.txt" % (j, typ), 'r')

            out_file = open('out_%i_%s.txt' % (j, typ), 'w')
            
            tgen = trigger_gen(trig_file)

            try:
                frame_index, trigger_counter, mask, E2 = tgen.next()
               
            # voor k40 die empty zijn
            except StopIteration:
                frame_index, trigger_counter, mask, E2 = -1, -1, -1, -1

            for root_index, passed, E in root_gen(root_file):
                e += 1
                total_events += 1

                if root_index == frame_index:
                    out_file.write("%i, %i, %i, %i, %i\n" % (root_index, passed, mask, E, E2))

                    total_trigger += 1
                    t += 1

                    try:
                        frame_index, trigger_counter, mask, E2 = tgen.next()
                    except StopIteration:
                        pass
                else:
                    out_file.write("%i, %i, %i, %i, %i\n" % (root_index, passed, 0, E, 0))

            print '\t', typ, j, e, t

            out_file.close()
            root_file.close()
            trig_file.close()
    print 'total events & triggered', total_events, total_trigger

def concatenate_files():
    print "\nconcatercate files"
    total_events = 0
    total_passed = 0
    total_triggered = 0
    final_out = open('final_out.txt', 'w')
    for j in [13, 14, 15]:
        for typ in EVT_TYPES:

            fpassed = 0
            ftriggered = 0
            fevent = 0
            out_file = open('out_%i_%s.txt' % (j, typ), 'r')

            for line in out_file:
                index, passed, mask, E, E2 = line.split(',')

                fevent += 1
                if int(passed) == 1:
                    fpassed += 1
                if int(mask) != 0:
                    ftriggered += 1

                final_out.write(line)

            out_file.close()

            total_events += fevent
            total_passed += fpassed
            total_triggered += ftriggered

            print '\t', typ, j, fevent, fpassed, ftriggered
    
    print 'total events & passed & triggered', total_events, total_passed, total_triggered
    final_out.close()


def write_to_hdf5():
    print "\nwrite to hdf5"
    final_out = open('final_out.txt', 'r')
    a = set()
    with h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5', 'a') as hfile:

        #dset_m = hfile.create_dataset("all_masks", dtype=int, shape=(NUM_TEST_EVENTS,)) 
        j = 0 
        for i, line in enumerate(final_out):
            
            index, passed, mask, E, E2 = line.split(',')

            if int(passed) == 1:
                hfile["all_masks"][j] = int(mask) 
                if hfile["all_num_hits"][j] != int(E):
                    print "ERROR ERROR ERRROR0"
                #if hfile["all_num_hits"][j] != int(E2):
                #    print "ERROR ERROR ERRROR1"
                #if int(E) != int(E2):
                #    print "ERROR ERROR ERRROR2"
                a.add(int(mask))
                j += 1
    print a
    final_out.close()

def make_histogram():
    final_out = open('final_out.txt', 'r')
    energies = []
    triggers = []
    for i, line in enumerate(final_out):
        index, passed, mask, E, E2 = line.split(',')
        
        energies.append(int(E))
        triggers.append(int(mask))

    num_events_nueCC_13 = 0

    f13 = open('out_13_nueCC.txt', 'r')
    for line in f13:
        num_events_nueCC_13 += 1
        
    energies = np.array(energies[:num_events_nueCC_13])
    triggers = np.array(triggers[:num_events_nueCC_13])
    fig, ax1 = plt.subplots(1,1)
    dens = False
    nok40 = np.where((energies != 0) & (energies < 1000))
    nok40_trg = np.where((energies != 0) & (energies < 1000) & (triggers != 0))
    he, be  = np.histogram(energies[nok40], bins=60, density=dens)
    het, _  = np.histogram(energies[nok40_trg], bins=be, range=(be.min(), be.max()), density=dens)

    ax1.plot(be[:-1], het / he.astype(float), drawstyle='steps') 
    ax1.set_ylim(0,1)
    ax1.set_xlim(0, 1000)
    ax1.legend()
    ax1.set_title('Trigger Efficientcy')
    ax1.set_ylabel('Fraction of events triggered')
    plt.show()

def main():
    #root_converter()
    #trigger_converter()
    #combine_trigger_and_root()
    #concatenate_files()
    write_to_hdf5()
    #make_histogram()

if __name__ == "__main__":
    main()
