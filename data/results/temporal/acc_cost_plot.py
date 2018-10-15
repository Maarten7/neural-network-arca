import matplotlib.pyplot as plt
fh = open('epoch_cost_acc.txt')

costs, accs = [], []
for line in fh:
    try:
        epoch, cost, acc = line.split(',')
    except:
        epoch, cost, acc, batch = line.split(',')
    
    costs.append(float(cost))
    accs.append(float(acc))

print min(costs)
plt.plot(costs)
plt.show()
plt.plot(accs)
plt.show()
