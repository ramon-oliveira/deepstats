import matplotlib.pyplot as plt

from keras import backend as K
from keras.engine.topology import Layer
import scipy.stats
import numpy as np
import datasets
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout, ELU
import time
import pandas as pd
import seaborn

# http://jmlr.org/proceedings/papers/v37/blundell15.pdf


class Bayesian(Layer):

    def __init__(self, output_dim, mean_prior, std_prior, **kwargs):
        self.output_dim = output_dim
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(Bayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.W = K.random_normal([input_shape[0]]+shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        self.W_sample = self.W*K.log(1.0 + K.exp(self.log_std)) + self.mean
        return K.batch_dot(x, self.W_sample) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class PoorBayesian(Layer):

    def __init__(self, output_dim, mean_prior, std_prior, **kwargs):
        self.output_dim = output_dim
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(PoorBayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.W = K.random_normal(shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        self.W_sample = self.W*K.log(1.0 + K.exp(self.log_std)) + self.mean
        return K.dot(x, self.W_sample) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class MyDropout(Layer):

    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(MyDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.dropout(x, level=self.p)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(MyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiasLayer(Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        init = np.zeros(input_dim)
        self.bias = K.variable(init)
        self.trainable_weights = []#[self.bias]

    def call(self, x, mask=None):
        return x + self.bias

    def get_output_shape_for(self, input_shape):
        return input_shape


def log_gaussian(x, mean, std):
    return -K.log(2*np.pi)/2.0 - K.log(std) - (x-mean)**2/(2*std**2)

def log_gaussian2(x, mean, log_std):
    log_var = 2*log_std
    return -K.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*K.exp(log_var))


def bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs):
    def loss(y_true, y_pred):
        log_p = K.variable(0.0)
        log_q = K.variable(0.0)
        nb_samples = batch_size
        for layer in model.layers:
            if type(layer) is Bayesian:
                mean = layer.mean
                log_std = layer.log_std
                W_sample = layer.W_sample
                # prior
                log_p += K.sum(log_gaussian(W_sample, mean_prior, std_prior))/nb_samples
                # posterior
                log_q += K.sum(log_gaussian2(W_sample, mean, log_std))/nb_samples

        #log_likelihood = objectives.categorical_crossentropy(y_true, y_pred)
        log_likelihood = K.sum(log_gaussian(y_true, y_pred, std_prior))

        return K.sum((log_q - log_p)/nb_batchs - log_likelihood)/batch_size
    return loss


def model_test(model, batch_size, X_test, y_test, labels_to_test):
    cnt = 0
    acc = 0
    for x, y in zip(X_test, y_test):
        if y in labels_to_test:
            cnt += 1
            probs = model.predict(np.array([x]*batch_size), batch_size=batch_size)
            pred_mean = probs.mean(axis=0)
            o = np.argmax(pred_mean)
            if o == labels_to_test.index(y):
                acc += 1
    return acc/cnt

dataset = 'mnist'
experiment_name = 'test_mnist_4labels_2unknown_1.1'
inside_labels = list(range(10))
#unknown_labels = [7, 9]
#hidden_layers = [512, 512]
#
#
(X_train, y_train), (X_test, y_test) = datasets.load_data(dataset, inside_labels)
#
batch_size = 100
mean_prior = 0.0
std_prior = 0.05
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
nb_batchs = X_train.shape[0]//batch_size
#
#model = Sequential()
#model.add(Bayesian(hidden_layers[0], mean_prior, std_prior, batch_input_shape=[batch_size, in_dim]))
#model.add(ELU())
#for h in hidden_layers[1:]:
#  model.add(Bayesian(h, mean_prior, std_prior))
#  model.add(ELU())
#model.add(Bayesian(out_dim, mean_prior, std_prior))
#model.add(Activation('softmax'))



inputs = Input(batch_shape=[batch_size, in_dim])
#noise = BiasLayer(trainable=False)(inputs)
network = Bayesian(512, mean_prior, std_prior)(inputs)
network = Bayesian(512, mean_prior, std_prior)(network)
network = Bayesian(out_dim, mean_prior, std_prior)(network)
network = Activation('softmax')(network)
model = Model(input=inputs, output=network)

loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)
model.compile(loss=loss, optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size)

#model.load_weights('weights/mnist/adversarial.h5')