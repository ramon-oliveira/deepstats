import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializations
from keras import activations
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.utils.np_utils import conv_output_length, conv_input_length


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
        return K.batch_dot(x, self.W) + self.bias

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


class BayesianConvolution2D(Layer):

    def __init__(self, nb_filter, nb_row, nb_col, mean_prior, std_prior,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.mean_prior = mean_prior
        self.std_prior = std_prior
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(BayesianConvolution2D, self).__init__(**kwargs)

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

        # self.W = K.random_normal([input_shape[0]]+shape, mean=self.mean_prior, std=self.std_prior)
        self.epsilon = K.random_normal(self.W_shape, mean=self.mean_prior, std=self.std_prior)
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

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        output = K.conv2d(x, self.W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Convolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
