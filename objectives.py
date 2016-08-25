import numpy as np
from layers import Bayesian
import keras.backend as K
from keras import objectives

# KL divergence between normal and standard normal N(0, I)
def KL_standard_normal(mean, log_std):
    return (-1.0 - 2.0*log_std + mean**2.0 + 2.0*K.exp(log_std))/2.0

# Following "Weight Uncertainty in Neural Networks" by Blundell et al.
def bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs):
    def loss(y_true, y_pred):
        KL_prior_posterior = K.variable(0.0)
        for layer in model.layers:
            if type(layer) is Bayesian:
                mean = layer.mean
                log_std = layer.log_std
                KL_prior_posterior += K.sum(KL_standard_normal(mean, log_std))/batch_size

        # Classification
        log_likelihood = -objectives.categorical_crossentropy(y_true, y_pred)

        # Regression
        #log_likelihood = K.sum(log_gaussian(y_true, y_pred, std_prior))

        return K.sum(KL_prior_posterior/nb_batchs - log_likelihood)/batch_size
    return loss

def log_gaussian(x, mean, std):
    return -K.log(2*np.pi)/2.0 - K.log(std) - (x-mean)**2/(2*std**2)

def log_gaussian2(x, mean, log_std):
    log_var = 2*log_std
    return -K.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*K.exp(log_var))

# Same loss as the one above, but without the KL analytical form for normals
def old_bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs):
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

        # Classification
        log_likelihood = -objectives.categorical_crossentropy(y_true, y_pred)

        # Regression
        #log_likelihood = K.sum(log_gaussian(y_true, y_pred, std_prior))

        return K.sum((log_q - log_p)/nb_batchs - log_likelihood)/batch_size
    return loss
