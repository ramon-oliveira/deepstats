import scipy.stats
import numpy as np
import dataloader
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, ELU, MaxPooling2D, Flatten
from layers import Bayesian, PoorBayesian, ProbabilisticDropout, BayesianConvolution2D
from objectives import bayesian_loss, explicit_bayesian_loss
import time
import pandas as pd
from sklearn import metrics
from keras.datasets import cifar10 as data
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = data.load_data()

img_rows, img_cols = 32, 32
img_channels = 3
mean_prior = 0.0
std_prior = 0.05
nb_classes = 10
batch_size = 50
nb_batchs = X_train.shape[0]//batch_size

# normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(X_train.shape)



model = Sequential()
model.add(BayesianConvolution2D(32, 3, 3, mean_prior, std_prior, border_mode='valid',
                                batch_input_shape=[batch_size] + list(X_train.shape[1:])))
model.add(ELU())
model.add(BayesianConvolution2D(32, 3, 3, mean_prior, std_prior))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(BayesianConvolution2D(64, 3, 3, mean_prior, std_prior, border_mode='same'))
model.add(ELU())
model.add(BayesianConvolution2D(64, 3, 3, mean_prior, std_prior))
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Bayesian(512, mean_prior, std_prior))
model.add(ELU())
model.add(Bayesian(nb_classes, mean_prior, std_prior))
model.add(Activation('softmax'))
loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)

model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model.fit(X_train, Y_train, nb_epoch=100, batch_size=batch_size, validation_data=(X_test, Y_test))
end_time = time.time()

print(end_time - start_time)
