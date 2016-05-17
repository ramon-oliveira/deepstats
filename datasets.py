import numpy as np
from keras.utils import np_utils

def load_data(dataset, train_labels):
    if dataset == 'mnist':
        from keras.datasets import mnist as data
    elif dataset == 'cifar10':
        from keras.datasets import cifar10 as data

    (X_train, y_train), (X_test, y_test) = data.load_data()

    idxs_train = [i for i, l in enumerate(y_train) if l in train_labels]
    X_train = X_train[idxs_train]/255
    X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    y_train = y_train[idxs_train]
    train_labels.sort()
    label_key = {i: l for i, l in enumerate(train_labels)}
    y_train = np.array([label_key[l] for l in y_train])
    y_train = np_utils.to_categorical(y_train)

    X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))/255

    return (X_train, y_train) , (X_test, y_test)
