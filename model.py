from keras.models import Sequential
from keras.layers import Dense
#create model
model = Sequential()
train_dataset = np.load('datasets/images.npy') 
#one-hot encode target column

#vcheck that target column has been converted
n_images= len(train_dataset)
#add model layers
model.add(Dense(200, activation='relu', input_shape=(n_images,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='softmax'))