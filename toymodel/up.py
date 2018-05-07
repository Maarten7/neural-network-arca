import numpy as np
import random 
def up():
    event = np.zeros((5, 12, 14))
    i = random.randint(6, 11)
    j = random.randint(0, 13)
    k = 0
    event[k][i][j] = 1

    while i > 0:
        k += 1
        try:
            i -= 1
            s = random.randint(-1, 1)
            j += s
            event[k][i][j] = .5
        except IndexError:
            break
    return event, np.array([0,1])
