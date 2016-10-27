import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations
from keras import activations
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.layers import Convolution2D
from keras.utils.np_utils import conv_output_length, conv_input_length

import pdb


class Bayesian(Layer):

    """Bayesian layer based on paper http://jmlr.org/proceedings/papers/v37/blundell15.pdf"""

    def __init__(self, output_dim, mean_prior, std_prior, **kwargs):
        self.output_dim = output_dim
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(Bayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.epsilon = K.random_normal([input_shape[0]]+shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='mean')
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='log_std')
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]), name='bias')
        self.W = self.epsilon*K.log(1.0 + K.exp(self.log_std)) + self.mean

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        return K.squeeze(K.batch_dot(K.expand_dims(x, dim=1), self.W), axis=1) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class PoorBayesian(Layer):

    """Efficient Bayesian layer (only one weights sample per mini batch)"""

    def __init__(self, output_dim, mean_prior, std_prior, **kwargs):
        self.output_dim = output_dim
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(PoorBayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.epsilon = K.random_normal(shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='mean')
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='log_std')
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]), name='bias')
        self.W = self.epsilon*K.log(1.0 + K.exp(self.log_std)) + self.mean

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        return K.dot(x, self.W) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class ProbabilisticDropout(Layer):

    """Probabilistic Dropout (performs dropout at test time)"""

    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(ProbabilisticDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.dropout(x, level=self.p)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(ProbabilisticDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoorBayesianConvolution2D(Convolution2D):

    def __init__(self, mean_prior, std_prior, *args, **kwargs):
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        super(PoorBayesianConvolution2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        # self.W = self.init(self.W_shape, name='{}_W'.format(self.name))

        input_dim, output_dim = initializations.get_fans(self.W_shape)
        v = np.sqrt(6.0 / (input_dim + output_dim))
        values = np.random.uniform(low=-v, high=v, size=self.W_shape)
        self.mean = K.variable(values, name='mean')
        values = np.random.uniform(low=-v, high=v, size=self.W_shape)
        self.log_std = K.variable(values, name='log_std')


        self.epsilon = K.random_normal(self.W_shape,
                                       mean=self.mean_prior, std=self.std_prior)
        self.W = self.epsilon*K.log(1.0 + K.exp(self.log_std)) + self.mean

        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.mean, self.log_std, self.b]
        else:
            self.trainable_weights = [self.mean, self.log_std]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
