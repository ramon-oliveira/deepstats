import numpy as np
from layers import Bayesian
import keras.backend as K

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
