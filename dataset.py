import numpy as np
import random
from matplotlib import pyplot as plt

def get_dataset():
    images = np.load('datasets/images.npy')
    classesDataset = np.zeros(images.shape[0])
    classesDataset[:204] = 9
    classesDataset[204:409] = 0
    classesDataset[409:615] = 7
    classesDataset[615:822] = 6
    classesDataset[822:1028] = 1
    classesDataset[1028:1236] = 8
    classesDataset[1236:1443] = 4
    classesDataset[1443:1649] = 3
    classesDataset[1649:1855] = 2
    classesDataset[1855:] = 5

    separationIndex = int(len(images) * 0.8)
    trainDataset = images[:separationIndex]
    evalDataset = images[separationIndex:]
    trainClasses = classesDataset[:separationIndex]
    evalClasses = classesDataset[separationIndex:]

    #TODO : Shuffle the data

    return trainDataset, trainClasses, evalDataset, evalClasses
