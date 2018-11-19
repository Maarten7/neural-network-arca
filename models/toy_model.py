import numpy as np

num_mini_timeslices = 200

def get_random_xyz():
    x = np.random.randint(13)
    y = np.random.randint(13)
    z = np.random.randint(18)
    return x, y, z


def get_event_klass_label():
    event = np.zeros((num_mini_timeslices, 13, 13, 18, 3))
    klass = np.random.randint(3) 
    label = np.zeros(3)
    label[klass] = 1
    return event, klass, label
    

def get_next_xyz(x, y, z, label):
    if np.argmax(label) == 0:
        z = z + 1
    elif np.argmax(label) == 1:
        z = z - 1

    if np.random.randint(3) == 0:
        x = x + 1
    elif np.random.randint(3) == 2:
        x = x - 1

    if np.random.randint(3) == 0:
        y = y + 1
    elif np.random.randint(3) == 2:
        y = y - 1
       
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    x = 12 if 12 < x else x
    y = 12 if 12 < y else y
    return x, y, z

def get_start(label):
    j = np.random.randint(num_mini_timeslices-18)
    
    if np.argmax(label) == 0:
        z = 0
    elif np.argmax(label) == 1:
        z = 17
    else:
        z = 10

    x = np.random.randint(13)
    y = np.random.randint(13)
    
    return j, x, y, z

def fill_noise(event):
    for i in range(num_mini_timeslices):
        x, y, z = get_random_xyz()
        event[i, x, y, z, :] += .30
    return event

def fill_event(event, label):
    j, x, y, z = get_start(label)
    for i in range(18):
        event[j + i, x, y, z, :] += .30
        x, y, z = get_next_xyz(x, y, z, label)
    return event

def random_line():
    event, klass, label = get_event_klass_label()
    for _ in range(10):
        event = fill_event(event, label)
    event = fill_noise(event)

    return event, label

def fill_matrix():
    event, klass, label = get_event_klass_label()
    if klass == 0:
        event[:, :, :, :, :] = 1.
    if klass == 1:
        event[:, :, :, :, :] = 0.5
    if klass == 2:
        event[:, :, :, :, :] = 0.

    return event, label

def add_line():
    event, klass, label = get_event_klass_label()
    x, y, z = get_random_xyz()
    if klass == 0:
        event[:, x, y, :, :] = 1
    elif klass == 1:
        event[:, x, :, z, :] = 1
    elif klass == 2:
        event[:, :, y, z, :] = 1

    return event, label

def make_toy():
    return add_line()
