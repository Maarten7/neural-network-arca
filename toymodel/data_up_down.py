import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import random

def npe():
    return np.random.choice([1, 1, 1, 2, 2, 3])

def Tot(npe):
    threshold = .22
    offset = 23.3
    slope = 7.
    curvature = 3. 
    saturation = 2e2

    tot = 0
    if npe >= threshold:
        x = npe - threshold
        tot = offset + npe * slope
        if curvature <= 0:
            tot *= np.sqrt(x / npe)
        else:
            tot *= 1. - np.exp(-curvature * x)
        tot *= saturation / (tot + saturation)
    return tot

def smear_tot(tot):
    return np.random.normal(tot, 3)


def event(down):
    event = np.zeros((10, 12, 14, 1))
    i = random.randint(0, 11)
    j = random.randint(0, 13)
    k = 0
    event[k][i][j] = [smear_tot(Tot(npe()))]

    while i < 12:
        k += 1
        try:
            if down:
                i += 1
            else:
                i -= 1
            s = random.randint(-1, 1)
            j += s
            event[k][i][j] = [smear_tot(Tot(npe()))]

            #NOISE
            ii = random.randint(0, 11)
            jj = random.randint(0, 13)
            event[k][ii][jj] += smear_tot(Tot(1))

        except IndexError:
            break
    if down:
        return event, np.array([1,0])
    else:
        return event, np.array([0,1])

def show_event():
    fig = plt.figure()
    ev, l = event(down=False)
    ev = ev.reshape(10, 12, 14)
    ims = []
    k = np.zeros((12, 14))
    for e in ev: 
        k = e
        im = plt.imshow(k)
        ims.append([im]) 

    ani = animation.ArtistAnimation(fig, ims)
    plt.show()

events = []
labels = []
for i in range(8000):
    e, l = event(down=False)
    events.append(e)
    labels.append(l)
    e, l = event(down=True)
    events.append(e)
    labels.append(l)

tevents = []
tlabels = []
for i in range(2000):
    e, l = event(down=False)
    tevents.append(e)
    tlabels.append(l)
    e, l = event(down=True)
    tevents.append(e)
    tlabels.append(l)
