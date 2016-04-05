import numpy as np
from keras.datasets import mnist
#from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, MaskedLayer
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn

class MyDropout(MaskedLayer):
    def __init__(self, p, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.p = p

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            X = K.dropout(X, level=self.p)
        return X

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'p': self.p}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


np.random.seed(1337)
batch_size = 128
nb_classes = 2
nb_epoch = 5
rg = l2(l=1e-3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

indexes29 = [i for i, c in enumerate(y_train) if c >= 2]
test_indexes29 = [i for i, c in enumerate(y_test) if c >= 2]
indexes01 = [i for i, c in enumerate(y_train) if c < 2]
test_indexes01 = [i for i, c in enumerate(y_test) if c < 2]

s0 = sum(1 for c in y_test if c == 0)
print('test class 0:', s0, s0/len(test_indexes01))
s1 = sum(1 for c in y_test if c == 1)
print('test class 1:', s1, s1/len(test_indexes01))


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.delete(X_train, indexes29, axis=0)
y_train = np.delete(y_train, indexes29, axis=0)
#y_train = np_utils.to_categorical(y_train, nb_classes)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
model.add(MyDropout(0.5))
model.add(Dense(1, activation='sigmoid', W_regularizer=rg))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd)
T = 50


def evaluate(X, l=2, p=0.5):
    N = X_train.shape[0]
    probs = []
    for i in range(T):
        probs.append(model.predict(np.array(X)))
    pred_mean = np.mean(probs, axis=0)
    pred_variance = np.var(probs, axis=0)
    #tau = l**2 * (1 - p) / (2 * N * rg.l2)
    #pred_variance += tau**-1
    return pred_mean, pred_variance


def plot_class(cl):
    indexes = [i for i, c in enumerate(y_test) if c != cl]
    m, v = evaluate(np.delete(X_test, indexes, axis=0))
    plt.figure()
    plt.title('DIGIT ' + str(cl))
    plt.hist(m)
    plt.hist(v)
    plt.xlim(0, 1)
    plt.xlabel('Probabilities')

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          show_accuracy=True)

for i in range(10):
    plot_class(i)