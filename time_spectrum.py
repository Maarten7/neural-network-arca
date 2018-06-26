from ROOT import *
import aa
from helper_functions import *


EventFile.read_timeslices = True
i = 0

aa = []
bb = []
cc = []
dd = []
ee = []
ff = []

for root_file, evt_type in root_files(test=True):
    print root_file
    f = EventFile(root_file)
    f.use_three_index_for_mc_reading = True
    
    if i > 50000:
        break
    
    for evt in f:
        i += 1
        if i > 50000:
            break

        max_mc = 0
        min_mc = 1e9 
        for hit in evt.mc_hits:
            time = hit.t
            
            if time > max_mc:
                max_mc= time
            if time < min_mc:
                min_mc= time


        max_al = 0
        min_al = 1e9
        for hit in evt.hits:
            time = evt.getMCtime(hit)
            
            if time > max_al:
                max_al = time
            if time < min_al:
                min_al = time

        a = max_mc - min_mc
        b = max_al - min_al

        c = min_mc - min_al
        d = max_al - max_mc

        e = 0 - min_al
        f = min_mc - 0
        
        aa.append(a)
        bb.append(b)
        cc.append(c)
        dd.append(d)
        ee.append(e)
        ff.append(f)

plt.hist(aa, bins=100)
plt.title('time spane mc hits')
plt.savefig('ts mc hits')
plt.show()

plt.hist(bb, bins=100)
plt.title('time spane total')
plt.savefig('ts total')
plt.show()

plt.hist(cc, bins=100)
plt.title('time spance pre mc hits')
plt.savefig('ts pre mc')
plt.show()

plt.hist(dd, bins=100)
plt.title('time spance post mc hits')
plt.savefig('ts post mc')
plt.show()

plt.hist(ee, bins=100)
plt.title('time spance tot neutrino')
plt.savefig('ts tot neutrino')
plt.show()

plt.hist(ff, bins=100)
plt.title('time spance neutrino tot hits')
plt.savefig('ts neutrino tot hits')
plt.show()
