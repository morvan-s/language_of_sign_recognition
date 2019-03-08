import numpy as np
import random


images = np.load('datasets/images.npy')
classes = np.load('datasets/classes.npy')
images = images.reshape(images.shape[0], 1, 64, 64)

classesDataset = []
for c in classes:
    for i in range(0, len(c)):
        if c[i] == 1.:
            classesDataset.append(i)
            break

dataset = []


random.shuffle(dataset)

separationIndex = int(len(images) * 0.8)
trainDataset = images[:separationIndex]
evalDataset = images[separationIndex:]
trainClasses = images[:separationIndex]
evalClasses = images[separationIndex:]
