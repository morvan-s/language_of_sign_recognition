from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt
import main
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

#create model
model = Sequential()

X_train = main.trainDataset
Y_train = main.trainClasses
X_test = main.evalDataset
Y_test = main.evalClasses


image_gen  = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2)


model = Sequential()
model.add(Convolution2D(6, (3, 3), activation='relu', input_shape=(1,64,64), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Convolution2D(12, (3, 3), activation='relu', input_shape=(1,64,64), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Convolution2D(24, (3, 3), activation='relu', input_shape=(1,64,64), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Convolution2D(36, (3, 3), activation='relu', input_shape=(1,64,64), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

image_gen.fit(X_train, augment=True)


batch_size = 128

model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=batch_size),
          steps_per_epoch=  X_train.shape[0]//batch_size,
          epochs=30,
          verbose=1)
# 9. Fit model on training data
#model.fit(X_train, Y_train, batch_size=32, nb_epoch=30, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
