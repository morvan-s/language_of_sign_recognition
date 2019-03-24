from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import dataset

# Settings
EPOCHS = 50
BATCH_SIZE = 32
DATA_AUGMENTATION = True

# Create model structure
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(64, (2, 2), activation='relu', input_shape=(64, 64, 1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(128, (2, 2), activation='relu', input_shape=(64, 64, 1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Convolution2D(256, (2, 2), activation='relu', input_shape=(64, 64, 1), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Load pre-processed data
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Train model
if DATA_AUGMENTATION:
    # With data augmentation
    generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.9, 1.1],
    )
    model.fit_generator(
        generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=generator.flow(x_test, y_test, batch_size=BATCH_SIZE)
    )
else:
    # Without data augmentation
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )

# Evaluate trained model with testing data
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save trained model according to TensorFlow.js format
# import tensorflowjs as tfjs
# tfjs.converters.save_keras_model(model, 'demo/model')
