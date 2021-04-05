import matplotlib.pyplot as plt
import numpy as np

train = []
dev = []
with open('example_losses.txt', 'r') as f:
    for line in f:
        if line.startswith("Epoch"):
            continue
        line = line[line.find(':'):]
        train.append(float(line[2:8]))
        line = line[line.find("s:")+3:]
        dev.append(float(line[0:6]))

fig, ax = plt.subplots()
ax.plot(np.arange(250), np.array(train), label='train_loss')
ax.plot(np.arange(250), np.array(dev), label='dev_loss')
ax.set_xlabel('# of epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.grid()
plt.show()