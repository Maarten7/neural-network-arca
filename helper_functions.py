from ROOT import * 
import aa
import numpy as np
import sys
import importlib

PATH = "/user/postm/neural-network-arca/"

EVT_TYPES = ['nueCC', 'anueCC', 'nueNC', 'anueNC', 'numuCC', 'anumuCC', 'nuK40', 'anuK40', 'nuATM']
NUM_CLASSES = 4
NUM_MINI_TIMESLICES = 50 

def save_output(model_title, cost, acc=0, epoch=0, batch=0):
    """ writes accuracy and cost to file
        input acc, accuracy value to write to file
        input cost, cost value to write to file"""

    print "Epoch %s\tcost: %f\tacc: %f\tbatch: %i" % (epoch, cost, acc, batch)

    with open(PATH + 'results/%s/epoch_cost_acc_%i.txt' % (model_title, NUM_MINI_TIMESLICES), 'a') as f:
        f.write(str(epoch) + ',' + str(cost) + ',' + str(acc) + ',' + str(batch) + '\n')

def import_model(only_model=True):
    """ imports a python module from command line. 
        Also import debug mode default is False"""        
    model = sys.argv[1].replace('/', '.')[:-3]
    model = importlib.import_module(model)
    if only_model: return model
    try:
        debug = eval(sys.argv[2])
        return model, debug
    except IndexError:
        return model, False

def random_event(k40=False):
    """ return a random evt object from data set"""
    # random file by random number and evt type
    i         = np.random.randint(1,16)
    evt_types = EVT_TYPES if k40 else EVT_TYPES[:6] 
    evt_type  = np.random.choice(evt_types)
    
    # format path string
    path = PATH + 'data/root_files'
    path += '/out_JEW_{0}_{1}.evt.root'
    n = path.format(evt_type, i)
    print n

    # open with aa / root
    EventFile.read_timeslices = True 
    f = EventFile(n)
    f.use_tree_index_for_mc_reading = True

    # random evt from file
    index = np.random.randint(0, len(f))
    f.begin()
    f.set_index(index)
    print f.index
    return f.evt


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
            yield n, evt_type

def extra_root_files():
    J = "JEW"
    path = PATH + 'data/root_files'
    path += '/out_{2}_{0}_{1}.root'
    yield path.format('numuCC', 16, J), 'numuCC'
    yield path.format('anumuCC', 16, J), 'anumuCC'
    yield path.format('nuK40', 16, J), 'nuK40' 
    yield path.format('anuK40', 16, J), 'anuK40'

    yield path.format('numuCC', 17, J), 'numuCC'
    yield path.format('anumuCC', 17, J), 'anumuCC'
    yield path.format('nuK40', 17, J), 'nuK40'
    yield path.format('anuK40', 17, J), 'anuK40'

    yield path.format('numuCC', 18, J), 'numuCC'
    yield path.format('anumuCC', 18, J), 'anumuCC'
    yield path.format('nuK40', 18, J), 'nuK40'

    yield path.format('numuCC', 19, J), 'numuCC'

def root_files_train():
    for rf, evt_type in root_files(range(1, 13)):
        yield rf, evt_type
    for rf, evt_type in extra_root_files():
        yield rf, evt_type

def root_files_test():
    for rf, evt_type in root_files(range(13, 16)):
        yield rf, evt_type
        

def pmt_id_to_dom_id(pmt_id):
    #channel_id = (pmt_id - 1) % 31
    dom_id     = (pmt_id - 1) / 31 + 1
    return dom_id

def doms_hit_pass_threshold(mc_hits, threshold, pass_k40):
    """ checks if there a at least <<threshold>> doms
        hit by monte carlo hits. retuns true or false"""
    if threshold == 0: return True
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

def num_events(threshold, root_files_set):
    """ calculates number of events with 
        num dom hits> threshold"""
    tra = {typ: 0 for typ in EVT_TYPES}
    EventFile.read_timeslices = True
        
    for root_file, evt_type in root_files_set():
        print root_file        
       
        if evt_type not in ['anumuCC', 'numuCC']:
            continue

        f = EventFile(root_file)
        f.use_tree_index_for_mc_reading = True

        if threshold == 0:
            tra[evt_type] += len(f) 
            continue

        for evt in f:
            if good_event(evt, evt_type):
                tra[evt_type] += 1
    print tra
    return tra 

DIR_TRAIN_EVENTS = {'anueNC': 17483, 'numuCC': 47064, 'nueCC': 28866, 'nuATM': 93903, 'anumuCC': 46205, 'anueCC': 28990, 'nueNC': 19418, 'nuK40': 51465, 'anuK40': 48034}
DIR_TEST_EVENTS  = {'anueNC': 4438,  'numuCC': 8593,  'nueCC': 7145,  'nuATM': 23410, 'anumuCC': 9212,  'anueCC': 7124,  'nueNC': 4946,  'nuK40': 10293, 'anuK40': 10293}

#root extra = numuCC 11842, anumuCC 9245 
#k40 extra 5 * 3431  waarvan 2 voor anuK40


DIR_TRAIN_LABELS = {'shower': DIR_TRAIN_EVENTS['nueCC'] + DIR_TRAIN_EVENTS['anueCC'] + DIR_TRAIN_EVENTS['nueNC'] + DIR_TRAIN_EVENTS['anueNC'],
                    'track':  DIR_TRAIN_EVENTS['numuCC'] + DIR_TRAIN_EVENTS['anumuCC'],
                    'k40':    DIR_TRAIN_EVENTS['nuK40'] + DIR_TRAIN_EVENTS['anuK40'],
                    'atm':    DIR_TRAIN_EVENTS['nuATM'],
                    }

DIR_TEST_LABELS = {'shower': DIR_TEST_EVENTS['nueCC'] + DIR_TEST_EVENTS['anueCC'] + DIR_TEST_EVENTS['nueNC'] + DIR_TEST_EVENTS['anueNC'],
                    'track':  DIR_TEST_EVENTS['numuCC'] + DIR_TEST_EVENTS['anumuCC'],
                    'k40':    DIR_TEST_EVENTS['nuK40'] + DIR_TEST_EVENTS['anuK40'],
                    'atm':    DIR_TEST_EVENTS['nuATM'],
                    }

NUM_TRAIN_EVENTS = sum(DIR_TRAIN_EVENTS.values())
NUM_TEST_EVENTS = sum(DIR_TEST_EVENTS.values())
NUM_EVENTS = NUM_TRAIN_EVENTS + NUM_TEST_EVENTS

NUM_TEST_EVENTS_NO_TRESHOLD = 81908
NUM_VAL_EVENTS = 1000

print "\nEvents"
print NUM_EVENTS
print "\n#Train Events, #Test Events"
print NUM_TRAIN_EVENTS, NUM_TEST_EVENTS
print "\nClasses"
print DIR_TRAIN_LABELS
print DIR_TEST_LABELS
print '\n'
