from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

import tensorflowjs as tfjs
import numpy as np
import dataset
import cv2

#create model
model = Sequential()

X_train, Y_train, X_eval, Y_eval  = dataset.get_dataset()

image_gen  = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0/255,
        zoom_range=0.2)
unique, count= np.unique(Y_train, return_counts=True)

model = Sequential()
model.add(Convolution2D(10, (2, 2), activation='relu', input_shape=(64,64,1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(20, (2, 2), activation='relu', input_shape=(64,64,1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(30, (2, 2), activation='relu', input_shape=(64,64,1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(40, (2, 2), activation='relu', input_shape=(64,64,1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#image_gen.fit(X_train)


batch_size = 128

# model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=batch_size),
#           steps_per_epoch=  X_train.shape[0]//batch_size,
#           epochs=30,
#           verbose=1)
# 9. Fit model on training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=40, verbose=1)

score = model.evaluate(X_eval, Y_eval, verbose=0)
tfjs.converters.save_keras_model(model, 'demo/model3')
