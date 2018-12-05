import matplotlib.pyplot as plt
fh = open('epoch_cost_acc_80.txt')
#fh = open('epoch_cost_acc.txt')

epochs, costs, accs = [], [], []
for line in fh:
    try:
        epoch, cost, acc = line.split(',')
    except:
        epoch, cost, acc, batch = line.split(',')
   
    epochs.append(int(epoch))
    costs.append(float(cost))
    accs.append(float(acc))

print "minimal cost", min(costs)
print "maximal acc", max(accs) 
plt.plot(costs)
plt.xlabel("training step")
plt.ylabel("Cross entropy cost per event")
plt.title("Cost of validation per 100 batches of 15 event set of 500 random events")
plt.show()

plt.ylim(0,1)
plt.plot(accs)
plt.ylabel("Accuracy")
plt.xlabel("training step")
plt.title("Acc of validation per 100 batches of 15 event set of 500 random events")
plt.show()

plt.plot(epochs)
plt.show()
