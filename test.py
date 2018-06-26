from ROOT import *
import aa
import numpy as np
import sys
import h5py
import importlib
from helper_functions import *

title = 'temporal'

f = h5py.File(PATH + 'data/hdf5_files/tbin50_all_events_labels_meta_%s.hdf5' % title, 'r')

lenz = 400 
spec = np.zeros(lenz)
spec_e = np.zeros(lenz)
spec_m = np.zeros(lenz)
spec_k = np.zeros(lenz)


maxx = 0
minn = 400
for i, bins in enumerate(f['all_bins']):
    lent = bins[0][-1]
    clss = np.argmax(f['all_labels'][i])

    if lent > maxx:
        maxx = lent
        print maxx
    if lent < minn:
        minn = lent
        print lent

    spec[lent] += 1

    if clss == 0:
        spec_e[lent] += 1
    if clss == 1:
        spec_m[lent] += 1
    if clss == 2:
        spec_k[lent] += 1

    if i % 10000 == 0:
        print i

plt.plot(range(400), spec, ls='steps')
plt.savefig('full_timespec')
plt.show()
plt.close()

plt.plot(range(400), spec_e, ls='steps', label='electron')
plt.plot(range(400), spec_m, ls='steps', label='muon')
plt.plot(range(400), spec_k, ls='steps', label='k40')
plt.legend()
plt.savefig('timespec')
plt.show()


print maxx
print minn
print max(np.nonzero(spec)[0])
print min(np.nonzero(spec)[0])

