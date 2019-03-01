from keras.models import Sequential
from keras.layers import Dense
#create model
model = Sequential()

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))