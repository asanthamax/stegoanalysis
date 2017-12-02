from envs.tensorflow.Lib.datetime import time
from keras import backend as k
from keras.callbacks import EarlyStopping, TensorBoard
from keras.constraints import maxnorm
from keras.layers import Convolution2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from sklearn.model_selection import train_test_split

from datapreprocessor import Datapreprocessor

input_placeholder = k.placeholder((1, 3, 200, 200))
first_layer = Convolution2D(32, 3, 3, input_shape=(3, 200, 200), border_mode='same', activation='relu', W_constraint=maxnorm(3))
first_layer.input = input_placeholder
model = Sequential()
model.add(first_layer)
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Flatten())
model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(1, activation='softmax'))

epochs = 100
learning_rate = 0.001
decay = learning_rate / epochs
sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
rms = RMSprop(lr=learning_rate, decay=decay)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

preprocess = Datapreprocessor('data/general', 'data/prepared')
dataset, labels = preprocess.preprepare_data()

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=10)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='auto')
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=10, batch_size=5, write_images=True, embeddings_freq=10)

history = model.fit(x_train, y_train, batch_size=32, nb_epoch=500, callbacks=[earlyStopping, tensorboard])

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))