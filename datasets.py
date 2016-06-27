import numpy as np
from keras.utils import np_utils

def load_data(dataset, inside_labels, unknown_labels):
    if dataset == 'mnist':
        from keras.datasets import mnist as data
    elif dataset == 'cifar':
        from keras.datasets import cifar10 as data

    (X_train, y_train), (X_test, y_test) = data.load_data()

    n_labels = len(inside_labels + unknown_labels)
    idxs_train = [i for i, l in enumerate(y_train) if l in inside_labels + unknown_labels]
    X_train = X_train[idxs_train]/255
    X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))

    y_train = y_train[idxs_train]
    y_train = y_train.reshape(y_train.shape[0])
    idxs_unknown = [i for i, l in enumerate(y_train) if l in unknown_labels]
    inside_labels.sort()
    label_key = {l: i for i, l in enumerate(inside_labels + unknown_labels)}
    y_train = np.array([label_key[l] for l in y_train])
    y_train = np_utils.to_categorical(y_train)
    y_train[idxs_unknown] = np.array([1.0/n_labels]*n_labels)

    X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))/255
    y_test = y_test.reshape(y_test.shape[0])

    return (X_train, y_train) , (X_test, y_test)
