from keras import backend as K
from keras.engine.topology import Layer
import scipy.stats
import numpy as np
import datasets
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, ELU
import time
import pandas as pd

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


def anomaly(name, network, dataset, inside_labels, nb_epochs, batch_size,
            hidden_layers=[512, 512], dropout_p=0.5):
    """

    # Arguments
        name: experiment name
        dataset: [mnist, cifar10]
        network: [bayesian, mlp-dropout, mlp]
    """
    assert dataset in ['mnist', 'cifar10']
    assert network in ['poor-bayesian', 'bayesian', 'mlp-dropout', 'mlp']
    assert len(hidden_layers) >= 1
    assert len(inside_labels) >= 2

    print('#'*50)
    print('#'*50)
    print('Experiment:', name)
    print('Network:', network)
    print('Dataset:', dataset)
    print('Inside Labels:', str(inside_labels))
    print('Number of epochs:', nb_epochs)
    print('Batch size:', batch_size)
    print('-'*50)

    (X_train, y_train), (X_test, y_test) = datasets.load_data(dataset, inside_labels)

    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1]
    nb_batchs = X_train.shape[0]//batch_size

    mean_prior = 0.0
    std_prior = 0.05

    model = Sequential()
    if network == 'bayesian':
        model.add(Bayesian(hidden_layers[0], mean_prior, std_prior, batch_input_shape=[batch_size, in_dim]))
        model.add(ELU())
        for h in hidden_layers[1:]:
            model.add(Bayesian(h, mean_prior, std_prior))
            model.add(ELU())
        model.add(Bayesian(out_dim, mean_prior, std_prior))
        model.add(Activation('softmax'))
        loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)
    elif network == 'poor-bayesian':
        model.add(PoorBayesian(hidden_layers[0], mean_prior, std_prior, input_shape=[in_dim]))
        model.add(ELU())
        for h in hidden_layers[1:]:
            model.add(PoorBayesian(h, mean_prior, std_prior))
            model.add(ELU())
        model.add(PoorBayesian(out_dim, mean_prior, std_prior))
        model.add(Activation('softmax'))
        loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)
    elif network == 'mlp-dropout':
        model.add(Dense(hidden_layers[0], input_shape=[in_dim]))
        model.add(ELU())
        for h in hidden_layers[1:]:
            model.add(Dense(h))
            model.add(ELU())
            model.add(MyDropout(dropout_p))
        model.add(Dense(out_dim))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'
    elif network == 'mlp-deterministic':
        model.add(Dense(hidden_layers[0], input_shape=[in_dim]))
        model.add(ELU())
        for h in hidden_layers[1:]:
            model.add(Dense(h))
            model.add(ELU())
            model.add(Dropout(dropout_p))
        model.add(Dense(out_dim))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'

    model.compile(loss=loss, optimizer='adadelta', metrics=['accuracy'])
    mod = X_train.shape[0]%batch_size
    if mod:
        X_train = X_train[:-mod]
        y_train = y_train[:-mod]
    start_time = time.time()
    model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size)
    end_time = time.time()

    test_pred_std = {x:[] for x in range(10)}
    test_entropy_bayesian = {x:[] for x in range(10)}

    cnt_in = 0
    acc_in = 0

    for i, (x, y) in enumerate(zip(X_test, y_test)):
        if network == 'poor-bayesian':
            probs = model.predict(np.array([x]*batch_size), batch_size=1)
        else:
            probs = model.predict(np.array([x]*batch_size), batch_size=batch_size)
        pred_mean = probs.mean(axis=0)
        pred_std = probs.std(axis=0)
        entropy = scipy.stats.entropy(pred_mean)

        test_pred_std[y].append(pred_std.mean())
        test_entropy_bayesian[y].append(entropy)

        if y in inside_labels:
            o = np.argmax(pred_mean)
            cnt_in += 1
            if o == inside_labels.index(y):
                acc_in += 1


    # Anomaly detection
    # by classical prediction entropy
    def anomaly_detection(anomaly_score_dict, metric_name, df):
        threshold = np.logspace(-10.0, 1.0, 1000)
        acc = {}
        for t in threshold:
            tp = 0.0
            tn = 0.0
            for l in anomaly_score_dict:
                if l in inside_labels:
                    tp += (np.array(anomaly_score_dict[l]) < t).mean()
                else:
                    tn += (np.array(anomaly_score_dict[l]) >= t).mean()
            tp /= len(inside_labels)
            tn /= 10.0 - len(inside_labels)
            bal_acc = (tp + tn)/2.0
            f1_score = 2.0*tp/(2.0 + tp - tn)
            acc[t] = [bal_acc, f1_score, tp, tn]

        print("{}\tscore\tthreshold\tTP\tTN".format(metric_name))
        sorted_acc = sorted(acc.items(), key= lambda x : x[1][0], reverse = True)
        print("\tbalanced acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        df.set_value(name, metric_name + " bal_acc", sorted_acc[0][1][0])
        df.set_value(name, metric_name + " bal_acc_threshold", sorted_acc[0][0])

        sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
        print("\tf1 score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        df.set_value(name, metric_name + ' f1_score', sorted_acc[0][1][1])
        df.set_value(name, metric_name + ' f1_score_threshold', sorted_acc[0][0])

    print('-'*50)
    df = pd.DataFrame()
    df.set_value(name, "train_time", end_time - start_time)
    df.set_value(name, "dataset", dataset)
    df.set_value(name, "test_acc", acc_in/cnt_in)
    df.set_value(name, "inside_labels", str(inside_labels))
    df.set_value(name, "nb_epochs", nb_epochs)
    anomaly_detection(test_pred_std, "Bayesian prediction STD", df)
    anomaly_detection(test_entropy_bayesian, "Bayesian entropy", df)
    return df
