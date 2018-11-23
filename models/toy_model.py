import numpy as np

def get_random_xyz():
    x = np.random.randint(13)
    y = np.random.randint(13)
    z = np.random.randint(18)
    return x, y, z


def get_event_klass_label(num_mini_timeslices):
    event = np.zeros((num_mini_timeslices, 13, 13, 18, 3))
    klass = np.random.randint(3) 
    label = np.zeros(3)
    label[klass] = 1
    return event, klass, label
    

def get_start(klass, num_mini_timeslices):
    x, y, z = get_random_xyz()
    j = np.random.randint(num_mini_timeslices-18)
    
    if klass == 0:
        z = 0
    elif klass == 1:
        z = 17
    else:
        z = 10

    return j, x, y, z 


def fill_noise(event, num_mini_timeslices):
    for i in range(num_mini_timeslices):
        x, y, z = get_random_xyz()
        event[i, x, y, z, :] += .30
    return event


def get_next_weird_xyz(x, y, z, klass):
    if klass == 0:
        z = z + 1
    elif klass == 1:
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

def get_line_start(klass, num_mini_timeslices):
    j = np.random.randint(num_mini_timeslices-18)
    x, y, z = get_random_xyz()
    if klass == 0:
        x = 0
    elif klass == 1:
        y = 0
    elif klass == 2:
        z = 0
    return j, x, y, z
    

def fill_noise(event, num_mini_timeslices):
    for i in range(num_mini_timeslices):
        x, y, z = get_random_xyz()
        event[i, x, y, z, :] += .30
    return event


def fill_event(event, label, num_mini_timeslices):
    j, x, y, z = get_start(label, num_mini_timeslices)
    for i in range(18):
        event[j + i, x, y, z, :] += .30
        x, y, z = get_next_xyz(x, y, z, label)
    return event


def random_line(num_mini_timeslices):
    event, klass, label = get_event_klass_label(num_mini_timeslices)
    for _ in range(10):
        event = fill_event(event, label)
    event = fill_noise(event, num_mini_timeslices)
    

def get_next_line_xyz(x, y, z, klass):
    if klass == 0:
        x += 1
    elif klass == 1:
        y += 1
    else:
       z += 1 
    if x == 13: x = 0
    if y == 13: y = 0
    return x, y, z


def random_movement(n_lines, n_noise, num_mini_timeslices):
    event, klass, label = get_event_klass_label(num_mini_timeslices)
    for _ in range(n_lines):
        j, x, y, z = get_start(klass, num_mini_timeslices)
        for i in range(18):
            event[j + i, x, y, z, :] += .30
            x, y, z = get_next_weird_xyz(x, y, z, klass)
    for _ in range(n_noise):
        event = fill_noise(event, num_mini_timeslices)

    return event, label

def fill_matrix(num_mini_timeslices):
    event, klass, label = get_event_klass_label(num_mini_timeslices)
    if klass == 0:
        event[:, :, :, :, :] = 1.
    if klass == 1:
        event[:, :, :, :, :] = 0.5
    if klass == 2:
        event[:, :, :, :, :] = 0.

    return event, label


def add_line(num_mini_timeslices):
    event, klass, label = get_event_klass_label(num_mini_timeslices)
    x, y, z = get_random_xyz()
    if klass == 0:
        event[:, x, y, :, :] = 1.
    elif klass == 1:
        event[:, x, :, z, :] = 1.
    elif klass == 2:
        event[:, :, y, z, :] = 1.

    return event, label


def add_line_in_steps(n_lines, num_mini_timeslices):
    event, klass, label = get_event_klass_label(num_mini_timeslices)
    for _ in range(n_lines):
        j, x, y, z = get_line_start(klass, num_mini_timeslices)
        for i in range(18):
            event[j + i, x, y, z, :] = 1.
            x, y, z = get_next_line_xyz(x, y, z, klass)
    return event, label


def make_toy(num_mini_timeslices):
#    return add_line_in_steps(10, num_mini_timeslices)
#   return fill_matrix()     
#   return all_line_in_steps()
   return random_movement(25, 5, num_mini_timeslices)
