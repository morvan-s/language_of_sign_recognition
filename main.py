import numpy as np
import random
from keras.utils import np_utils

images = np.load('datasets/images.npy')
classes = np.load('datasets/classes.npy')
images = images.reshape(images.shape[0],1,64,64)
classesDataset = []
for c in classes:
    for i in range(0, len(c)):
        if c[i] == 1.:
            classesDataset.append(i)
            break

dataset = []
images.astype('float32')
images /= 255
classesDataset = np_utils.to_categorical(classesDataset, 10)

separationIndex = int(len(images) * 0.8)
trainDataset = images[:separationIndex]
evalDataset = images[separationIndex:]
trainClasses = classesDataset[:separationIndex]
evalClasses = classesDataset[separationIndex:]
