import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 16, 
    'font.family': 'serif',
    'pgf.rcfonts': True,
    'pgf.texsystem': "pdflatex",
    'figure.figsize': [8, 4], 
    'figure.autolayout': True,
    })

fh = open('epoch_cost_acc.txt')

costs, accs = [], []
for line in fh:
    try:
        epoch, cost, acc = line.split(',')
    except:
        epoch, cost, acc, batch = line.split(',')
    
    costs.append(float(cost))
    accs.append(float(acc))
print epoch

min_cost_y = min(costs)
min_cost_x = np.argmin(costs)
max_acc_y = accs[min_cost_x]


fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(costs)
ax1.plot(min_cost_x, min_cost_y, 'ro', label='Weights frozen')
ax1.set_xlabel("Training step")
ax1.set_ylabel("Cross entropy per event")
ax1.set_title("Cost")
ax1.legend()

ax2.plot(accs)
ax2.plot(min_cost_x, max_acc_y, 'ro')
ax2.set_ylim(0,1)
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Training step")
ax2.set_title("Accuracy")
fig.suptitle("Training process")
fig.savefig("trainingsteps.pgf")
plt.show()
