import numpy as np
import random

images = np.load('datasets/images.npy')
classes = np.load('datasets/classes.npy')

classesDataset = []
for c in classes:
    for i in range(0, len(c)):
        if c[i] == 1.:
            classesDataset.append(i)
            break

dataset = []
for i in range (0, len(images)):
    dataset.append((classesDataset[i], images[i]))

random.shuffle(dataset)

separationIndex = int(len(dataset) * 0.8)
trainDataset = dataset[:separationIndex]
evalDataset = dataset[separationIndex:]
