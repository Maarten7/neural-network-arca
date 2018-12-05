# jtrigger.py
# Maarten Post
""" Takes the test data set and looks what events are triggered by JTriggerEfficenty
    in order to compare it with KM3NNET"""
from ROOT import *
import aa
import numpy as np
import sys
import h5py

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


def trigger_converter():
    #EventFile.read_timeslices = True
    for JTP, typ in root_files(range(13, 16), 'JTP'):
        print JTP
        code = JTP.split('/')[-1].split('.')[0].split('_')[-1]
        with open('triggers_%s_%s.txt' % (code, typ), 'w') as tf:
            f = EventFile(JTP)
            #f.use_tree_index_for_mc_reading = False
            for evt in f:
                tf.write("%i, %i\n" % (evt.frame_index, evt.trigger_mask))

def root_converter():
    EventFile.read_timeslices = True
    k = 0
    for root_file, typ in root_files(range(13, 16), 'JEW'):
        print root_file
        code = root_file.split('/')[-1].split('.')[0].split('_')[-1]
        with open('roots_%s_%s.txt' % (code, typ), 'a') as tf:
            f = EventFile(root_file)
            f.use_tree_index_for_mc_reading = True
            for i, evt in enumerate(f):
                passes = doms_hit_pass_threshold(evt.mc_hits, 5, True)
                tf.write("%i, %i\n" % (i, passes))
                if passes:
                    k += 1

def loop():
    EventFile.read_timeslices = True
    for root_file, typ in root_files(range(13, 16)):
        print root_file
        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True
        for evt in f:
            print len(evt.hits), evt.hits.size()

def trigger_gen(triggerfile):
    for line in triggerfile:
        line = line.split(',')
        frame_index, mask = line
        yield int(frame_index), int(mask)

def root_gen(rootfile):
    for line in rootfile:
        line = line.split(',')
        frame_index, passed = line
        yield int(frame_index), int(passed)

def combine_trigger_and_root():
    for j in [13, 14, 15]:
        for typ in EVT_TYPES:
            root_file = open("roots_%i_%s.txt" % (j, typ), 'r')
            trig_file = open("triggers_%i_%s.txt" % (j, typ), 'r')
            out_file = open('out_%i_%s.txt' % (j, typ), 'w')
            
            tgen = trigger_gen(trig_file)

            try:
                trig_index, mask = tgen.next()
            except StopIteration:
                trig_index, mask = -1, -1

            for root_index, passed in root_gen(root_file):

                if root_index == trig_index:
                    out_file.write("%i, %i, %i\n" % (root_index, passed, mask))
                    try:
                        trig_index, mask = tgen.next()
                    except StopIteration:
                        pass
                else:
                    out_file.write("%i, %i, %i\n" % (root_index, passed, 0))
                    pass

            out_file.close()
            root_file.close()
            trig_file.close()

def concatenate_files():
    final_out = open('final_out.txt', 'w')
    for j in [13, 14, 15]:
        for typ in EVT_TYPES:
            k = 0
            out_file = open('out_%i_%s.txt' % (j, typ), 'r')

            for line in out_file:
                index, passed, mask = line.split(',')
                if int(passed) == 1:
                    k += 1
                final_out.write(line)

            out_file.close()

            print j, typ, k

    final_out.close()

def count_passed_and_triggered(final_out_file):
    tp, tt, to = 0, 0, 0
    passed_and_triggered = 0
    not_passed_but_triggered = 0
    final_out = open(final_out_file, 'r')

        
    for line in final_out:
        index, passed, mask = line.split(',')

        to += 1
        if int(passed) == 1:
            tp += 1
        if int(mask) > 0:
            tt += 1
        if int(passed) == 0 and int(mask) != 0:
            not_passed_but_triggered += 1
        if int(passed) == 1 and int(mask) != 0:
            passed_and_triggered += 1 

    print final_out_file
    print 'total passed:\t', tp
    print 'total triggered\t', tt
    print 'total events\t', to
    print 'not passed but triggered', not_passed_but_triggered 
    print 'passed_and_triggered', passed_and_triggered 
    print ''
    final_out.close()


def write_to_hdf5():
    final_out = open('final_out.txt', 'r')
    a = set()
    with h5py.File(PATH + 'data/hdf5_files/20000ns_400ns_all_events_labels_meta_test.hdf5', 'a') as hfile:

        #dset_m = hfile.create_dataset("all_masks", dtype=int, shape=(NUM_TEST_EVENTS,)) 
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
    #trigger_converter()
    #root_converter()
    combine_trigger_and_root()
    concatenate_files()
    write_to_hdf5()

if __name__ == "__main__":
    main()
