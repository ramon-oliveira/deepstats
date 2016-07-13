import numpy as np
from keras.utils import np_utils
from collections import defaultdict

def load(dataset, inside_labels, unknown_labels, with_unknown):
    if dataset == 'mnist':
        from keras.datasets import mnist as data
    elif dataset == 'cifar':
        from keras.datasets import cifar10 as data

    inside_labels.sort()
    unknown_labels.sort()

    (X_train, y_train), (X_test, y_test) = data.load_data()

    idxs_labels = defaultdict(list)
    for i, label in enumerate(y_train):
        idxs_labels[label].append(i)

    idxs_train = []
    if with_unknown:
        total = sum(len(idxs_labels[label]) for label in idxs_labels if label in inside_labels)
        per_labels = total//(len(inside_labels) + len(unknown_labels))
        for label in idxs_labels:
            if label in inside_labels + unknown_labels:
                idxs_train += idxs_labels[label][:per_labels]
    else:
        for label in idxs_labels:
            if label in inside_labels:
                idxs_train += idxs_labels[label]

    X_train = X_train[idxs_train]/255
    X_train = X_train.reshape(X_train.shape[0], -1)

    y_train = y_train[idxs_train]
    y_train = y_train.reshape(y_train.shape[0])

    idxs_inside = [i for i, l in enumerate(y_train) if l in inside_labels]
    idxs_unknown = [i for i, l in enumerate(y_train) if l in unknown_labels]

    label_key = {l: i for i, l in enumerate(inside_labels + unknown_labels)}
    y_train_int = np.array([label_key[l] for l in y_train])
    y_train = np.zeros([len(y_train_int), len(inside_labels)])
    y_train[idxs_inside, :] = np_utils.to_categorical(y_train_int[idxs_inside])
    y_train[idxs_unknown, :] = np.array([1.0/len(inside_labels)]*len(inside_labels))

    X_test = X_test.reshape(X_test.shape[0], -1)/255
    y_test = y_test.reshape(y_test.shape[0])

    assert y_train.shape[1] == len(inside_labels)

    return (X_train, y_train) , (X_test, y_test)
