import matplotlib.pyplot as plt
import numpy as np
fh = open('epoch_cost_acc_50.txt')
#fh = open('epoch_cost_acc_50_.txt')

epochs, costs, accs, batchs = [], [], [], []
for line in fh:
    epoch, cost, acc, batch = line.split(',')
    epochs.append(int(epoch))
    costs.append(float(cost))
    accs.append(float(acc))
    batchs.append(float(batch))

print "minimal cost", min(costs), np.argmin(costs)
print "maximal acc", max(accs), np.argmax(accs)
print "num", len(accs) 

plt.plot(costs, 'b')
plt.xlabel("training step")
plt.ylabel("Cross entropy cost per event")
plt.title("Cost of validation per 100 batches of 15 event set of 500 random events")
plt.show()

plt.ylim(0,1)
plt.plot(accs, 'b')
plt.ylabel("Accuracy")
plt.xlabel("training step")
plt.title("Acc of validation per 100 batches of 15 event set of 500 random events")
plt.show()

plt.plot(epochs)
plt.show()
