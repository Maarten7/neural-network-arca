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

EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40', 'nuATM']
NUM_CLASSES = 4
NUM_MINI_TIMESLICES = 50 

def pmt_id_to_dom_id(pmt_id):
    #channel_id = (pmt_id - 1) % 31
    dom_id     = (pmt_id - 1) / 31 + 1
    return dom_id

def make_file_str(evt_type, i, J='JEW'):
    """ returns a file str of evt_type root files"""
    i = str(i)
    path = PATH + 'data/root_files'
    path += '/out_{2}_{0}_{1}.root'
    n = path.format(evt_type, i, J)
    return n

def root_files(range, J='JEW'):
    """ outputs strings of all root_files"""
    for i in range:
        for evt_type in EVT_TYPES:
            n = make_file_str(evt_type, i, J=J)
            yield n, evt_type, i

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

def good_event(evt, evt_type):
    if evt_type in ['nuK40', 'anuK40']:
        return True 
    elif evt_type == 'nuATM':
        return len(evt.mc_trks) > 2 and doms_hit_pass_threshold(evt.mc_hits, 5, 0)
    else:
        return doms_hit_pass_threshold(evt.mc_hits, 5, 0) 

def root_converter():
    print "\nroot converter"
    EventFile.read_timeslices = True
    total_passed = 0
    total  = 0
    for root_file, typ, j in root_files(range(13, 16), 'JEW'):
        print '\t', root_file

        with open('roots_%s_%s.txt' % (typ, j), 'w') as tf:
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True

            total += len(f)

            passed_in_file = 0
            for i, evt in enumerate(f):

                passes = good_event(evt, typ)
                tf.write("%i, %i\n" % (evt.frame_index, passes))

                if passes:
                    total_passed   += 1
                    passed_in_file += 1

            print "\t", len(f), passed_in_file 

    print total, total_passed 

def root_gen(rootfile):
    for line in rootfile:
        line = line.split(',')
        frame_index, passed = line
        yield int(frame_index), int(passed)

def trigger_converter():
    print "\ntrigger converter"
    EventFile.read_timeslices = False 
    total_trigger_events = 0

    for JT, typ, j in root_files(range(13, 16), 'JTE'):
        print '\t', JT

        if typ != 'nuATM': continue

        with open('triggers_%s_%s.txt' % (typ, j), 'w') as tf:
            f = EventFile(JT)
            print '\t', typ, j, len(f)
            total_trigger_events += len(f)
            for evt in f:
                tf.write("%i, %i, %i\n" % (evt.frame_index, evt.trigger_counter, evt.trigger_mask))

    print 'total triggered', total_trigger_events

def trigger_gen(triggerfile):
    for line in triggerfile:
        frame_index, trigger_counter, mask = line.split(',')
        yield int(frame_index) - 1, int(trigger_counter) + 1, int(mask)

def combine_trigger_and_root():
    print "\ncombine trigger and root"

    total_trigger = 0
    total_events = 0
    for j in [13, 14, 15]:
        for typ in EVT_TYPES:
            triggered = 0
            events = 0

            root_file = open("roots_%s_%s.txt" % (typ, j), 'r')
            trig_file = open("triggers_%s_%s.txt" % (typ, j), 'r')

            out_file = open('out_%s_%s.txt' % (typ, j), 'w')
            
            tgen = trigger_gen(trig_file)

            try:
                frame_index, trigger_counter, mask  = tgen.next()
            except StopIteration:
                # voor k40 die empty zijn
                frame_index, trigger_counter, mask = -1, -1, -1

            for root_index, passed in root_gen(root_file):
                events += 1
                total_events += 1

                if root_index == frame_index:
                    out_file.write("%i, %i, %i\n" % (root_index, passed, mask))

                    total_trigger += 1
                    triggered += 1

                    try:
                        frame_index, trigger_counter, mask = tgen.next()
                    except StopIteration:
                        pass
                else:
                    out_file.write("%i, %i, %i\n" % (root_index, passed, 0))

            print '\t', typ, j, events, triggered

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
            out_file = open('out_%s_%s.txt' % (typ, j), 'r')

            for line in out_file:
                index, passed, mask = line.split(',')

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
    with h5py.File(PATH + 'data/hdf5_files/20000ns_250ns_all_events_labels_meta_test.hdf5', 'a') as hfile:

        try:
            dset_m = hfile.create_dataset("all_masks", dtype=int, shape=(62044,)) 
        except RuntimeError:
            pass

        j = 0 
        for i, line in enumerate(final_out):
            
            index, passed, mask = line.split(',')

            if int(passed) == 1:
                hfile["all_masks"][j] = int(mask) 
                a.add(int(mask))
                j += 1
    print a
    final_out.close()

def main():
    #root_converter()
    #trigger_converter()
    combine_trigger_and_root()
    concatenate_files()
    #write_to_hdf5()
    pass

if __name__ == "__main__":
    main()
