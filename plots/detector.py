import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.size': 16, 
    'font.family': 'serif',
    'pgf.rcfonts': True,
    'pgf.texsystem': "pdflatex",
    'figure.figsize': [12, 8], 
    'figure.autolayout': True,
    })

with open('detector', 'r') as d:
    i = 0
    fig, (ax, ax2) = plt.subplots(1,2)
    for line in d:
        x, y, z, pmt_id = line.split()
        if z == '711.961':
            i+=1
            ax.plot(float(x), float(y), 'b.')
            ax.text(float(x),float(y), str(i), fontsize=12)
    ax.plot([-219.15, 226.93], [-395.25, 394.7], linewidth=5, alpha=.5)
    ax.plot([-228.45, 234.36], [391, -391.5], linewidth=5, alpha=.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Arca detector (above view)')

lines = np.array([
    [0,   0,   109, 110, 111, 0,   0,  0,   0,   0,   0,   0,   0 ],
    [0,   81,  82,  83,  84,  85,  86, 0,   0,   0,   0,   0,   0 ],
    [108, 80,  53,  54,  55,  56,  57, 87,  112, 0,   0,   0,   0 ],
    [107, 79,  52,  31,  32,  33,  34, 58,  88,  113, 0,   0,   0 ],
    [106, 78,  51,  30,  15,  16,  17, 35,  59,  89,  114, 0,   0 ],
    [0,   77,  50,  29,  14,  5,   6,  18,  36,  60,  90,  115, 0 ],
    [0,   76,  49,  28,  13,  4,   1,  7,   19,  37,  61,  91,  0 ],
    [0,   105, 75,  48,  27,  12,  3,  2,   10,  24,  44,  70,  0 ],
    [0,   0,   104, 74,  47,  26,  11, 9,   8,   22,  42,  68,  99],
    [0,   0,   0,   103, 73,  46,  25, 23,  21,  20,  40,  66,  97],
    [0,   0,   0,   0,   102, 72,  45, 43,  41,  39,  38,  64,  95],
    [0,   0,   0,   0,   0,   101, 71, 69,  67,  65,  63,  62,  93],
    [0,   0,   0,   0,   0,   0,   0,  100, 98,  96,  94,  92,  0 ]
    ])

ax2.set_xlim((0,13))
ax2.set_ylim((0,13))
for i in range(13):
    for j in range(13):
        ax2.text(i, j, str(lines[j,i]), fontsize=12)
        if lines[j, i] != 0:
            ax2.plot(i, j, 'b.')
ax2.set_aspect('equal')
ax2.set_xlabel('x index')
ax2.set_ylabel('y index')
ax2.xaxis.set(ticks=range(0,13))
ax2.yaxis.set(ticks=range(0,13))
ax2.plot([6,6],[1,11], linewidth=5, alpha=.5)
ax2.plot([1,11],[6,6], linewidth=5, alpha=.5)
ax2.set_title('Arca matrix(above view)')
plt.savefig('detector.pgf')
plt.show()
    
with open('detector', 'r') as d:
    i = 0
    fig, ax = plt.subplots()
    for line in d:
        x, y, z, pmt_id = line.split()
        if z == '711.961':
            i+=1
            ax.plot(float(x), float(y), 'b.')
            ax.text(float(x),float(y), str(i), fontsize=12)
    ax.plot([-219.15, 226.93], [-395.25, 394.7], linewidth=5, alpha=.5)
    ax.plot([-228.45, 234.36], [391, -391.5], linewidth=5, alpha=.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Arca detector (above view)')
    plt.savefig('detector1.pgf')
    plt.show()
