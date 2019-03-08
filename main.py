import numpy as np
import random
from keras.utils import np_utils
from matplotlib import pyplot as plt

images = np.load('datasets/images.npy')
classesDataset = np.load('datasets/classes.npy')

print(classesDataset[300])
plt.subplot(5, 4, 1)
plt.axis('off')
plt.imshow(images[300].squeeze())
plt.show()

images = images.reshape(images.shape[0],64,64,1)

images.astype('float32')
images /= 255

separationIndex = int(len(images) * 0.8)
trainDataset = images[:separationIndex]
evalDataset = images[separationIndex:]
trainClasses = classesDataset[:separationIndex]
evalClasses = classesDataset[separationIndex:]
