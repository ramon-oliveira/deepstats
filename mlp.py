'''
For l = 0.01

---------- [0-1] ----------

            Bayesian                     Frequentist

mean        [0.46374818  0.53627337]     [0.46374818  0.53627337]
variance    [506600.246  506600.247]     [0.24626724  0.24630987]

---------- [2-9] ----------

            Bayesian                      Frequentist

mean        [0.62641634  0.38111495]      [0.62641634  0.38111495]
variance    [506600.131  506600.133]      [0.13127033  0.13332274]


###############################################################################

For l = 0.9

---------- [0-1] ----------

            Bayesian                      Frequentist

mean        [0.46374818    0.53627337]    [0.46374818  0.53627337]
variance    [62.78947711  62.78951975]    [0.24626724  0.24630987]

---------- [2-9] ----------

            Bayesian                      Frequentist

mean        [0.62641634  0.38111495]      [ 0.62641634  0.38111495]
variance    [62.6744802  62.6765326]      [ 0.13127033  0.13332274]

'''



import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils

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
y_train = np_utils.to_categorical(y_train, nb_classes)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid', W_regularizer=rg))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd)


def evaluate(probs, l=2, p=0.5):
    N = X_train.shape[0]
    pred_mean = np.mean(probs, axis=0)
    pred_variance = np.var(probs, axis=0)
    tau = l**2 * (1 - p) / (2 * N * rg.l2)
    pred_variance += tau**-1
    print('')
    print('\t\tBayesian\tFrequentist')
    print('')
    print('mean\t\t{0}\t\t{1}'.format(pred_mean, probs.mean(axis=0)))
    print('variance\t{0}\t\t{1}'.format(pred_variance, probs.var(axis=0)))


model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=10,
          show_accuracy=True)

print('-'*10, '[0-1]', '-'*10)
evaluate(model.predict(np.delete(X_test, test_indexes29, axis=0)))
print('-'*10, '[2-9]', '-'*10)
evaluate(model.predict(np.delete(X_test, test_indexes01, axis=0)))
