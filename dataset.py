import numpy as np
import random
from matplotlib import pyplot as plt
from keras.utils import np_utils

def load_data():
    # Load images
    images = np.load('datasets/images.npy')

    # Match images with their value
    classes = np.zeros(images.shape[0])
    classes[:204] = 9
    classes[204:409] = 0
    classes[409:615] = 7
    classes[615:822] = 6
    classes[822:1028] = 1
    classes[1028:1236] = 8
    classes[1236:1443] = 4
    classes[1443:1649] = 3
    classes[1649:1855] = 2
    classes[1855:] = 5

    # Reshape images to fit model
    classes = np_utils.to_categorical(classes, 10)
    images = images.reshape(images.shape[0], 64, 64, 1)

    # Shuffle dataset
    combined = list(zip(images, classes))
    random.shuffle(combined)
    images[:], classes[:] = zip(*combined)

    # Split dataset in two : 80% for training, 20% for tests
    splitIndex = int(len(images) * 0.8)
    x_train = images[:splitIndex]
    x_test = images[splitIndex:]
    y_train = classes[:splitIndex]
    y_test = classes[splitIndex:]
    
    return (x_train, y_train), (x_test, y_test)
