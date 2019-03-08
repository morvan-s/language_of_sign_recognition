from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt

#create model
model = Sequential()
X_train = np.load('X.npy')
y_train = np.load('Y.npy')
X_test= np.load('X.npy')
y_test= np.load('Y.npy')


n_cols = X_train.shape[1]


# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 9)
Y_test = np_utils.to_categorical(y_test, 9)

print(X_train.shape)
# 7. Define model architecture
model = Sequential()

#add model layers


model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
