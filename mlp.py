import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

np.random.seed(1337)
batch_size = 128
nb_classes = 10
nb_epoch = 5
rg = l2(l=1e-5)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

indexes = [i for i, c in enumerate(y_train) if c >= 2]
test_indexes = [i for i, c in enumerate(y_test) if c >= 2]

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.delete(X_train, indexes, axis=0)
y_train = np.delete(y_train, indexes, axis=0)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(512, activation='relu', W_regularizer=rg, input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', W_regularizer=rg))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd)


def evaluate(probs, l=0.9, p=0.5):
    N = X_train.shape[0]
    pred_mean = np.mean(probs, axis=0)
    pred_variance = np.var(probs, axis=0)
    tau = l**2 * (1 - p) / (2 * N * rg.l2)
    pred_variance += tau**-1
    print('')
    print('\t\tBayesian\tFrequentist')
    print('')
    print('mean\t\t{0:.2f}\t\t{1:.2f}'.format(pred_mean[0], probs.mean()))
    print('variance\t{0:.2f}\t\t{1:.2f}'.format(pred_variance[0], probs.var()))


for i in range(nb_epoch):
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=1,
              show_accuracy=True)
    out = model.predict(X_test)
    evaluate(out)
    score = model.evaluate(np.delete(X_test, test_indexes, axis=0),
                           np.delete(y_test, test_indexes, axis=0),
                           show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
