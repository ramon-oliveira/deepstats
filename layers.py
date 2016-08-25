import numpy as np
from keras import backend as K
from keras.engine.topology import Layer


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
        self.W = K.random_normal([input_shape[0]]+shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='mean')
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='log_std')
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]), name='bias')

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        self.W_sample = self.W*K.log(1.0 + K.exp(self.log_std)) + self.mean
        return K.batch_dot(x, self.W_sample) + self.bias

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
        self.W = K.random_normal(shape, mean=self.mean_prior, std=self.std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='mean')
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape), name='log_std')
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]), name='bias')

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        self.W_sample = self.W*K.log(1.0 + K.exp(self.log_std)) + self.mean
        return K.dot(x, self.W_sample) + self.bias

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
