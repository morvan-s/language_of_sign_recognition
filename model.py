from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt
import main
#create model
model = Sequential()

X_train = main.trainDataset
Y_train = main.trainClasses
X_test = main.evalDataset
Y_test = main.evalClasses


model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(1,64,64)))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))

model.add(Dense(500, activation='relu'))
model.add(Dense(64, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=30, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
