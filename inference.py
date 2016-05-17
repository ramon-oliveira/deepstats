from keras import backend as K
from keras.engine.topology import Layer
import scipy.stats
import numpy as np
import datasets
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import objectives
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


class Bayesian(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Bayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.W = K.random_normal(shape)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.stdev = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.stdev, self.bias]

    def call(self, x, mask=None):
        return K.dot(x, self.W*K.exp(self.stdev) + self.mean) + self.bias

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


class Dropout(Layer):

    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(Dropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.dropout(x, level=self.p)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



batch_size = 1
hidden = 512
dataset = 'mnist'
network = 'mlp'
train = False
train_labels = [0, 1]
(X_train, y_train), (X_test, y_test) = datasets.load_data(dataset, train_labels)

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
nb_batchs = X_train.shape[0]//batch_size

def bayesian_loss(y_true, y_pred):
    ce = objectives.categorical_crossentropy(y_true, y_pred)
    kl = K.variable(0.0)
    for layer in model.layers:
        if type(layer) is Bayesian:
            mean = layer.mean
            stdev = layer.stdev
            kl = kl + K.sum((mean**2)/2.0 + (K.exp(2*stdev) - 1.0 - 2*stdev)/2.0)
    return ce + kl/nb_batchs

model = Sequential()
if network == 'bayesian':
    model.add(Bayesian(hidden, input_shape=[in_dim]))
    model.add(Activation('relu'))
    model.add(Bayesian(out_dim))
    model.add(Activation('softmax'))
    loss = bayesian_loss
elif network == 'mlp':
    model.add(Dense(hidden, input_shape=[in_dim]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('softmax'))
    loss = 'categorical_crossentropy'

optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-3, nesterov=True)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
if train:
    mc = ModelCheckpoint(network+"-best-weights.h5", monitor='val_acc', 
                         save_best_only=True)
    model.fit(X_train, y_train, nb_epoch=20, batch_size=batch_size, 
              validation_split=0.2, callbacks=[mc])
else:
    model.load_weights(network+'-best-weights.h5')


test_pred_mean = {x:[] for x in range(10)}
test_pred_std = {x:[] for x in range(10)}
test_entropy_bayesian = {x:[] for x in range(10)}
test_variation_ratio = {x:[] for x in range(10)}

for i, x in enumerate(X_test[:100]):
    probs = model.predict(np.array([x]*50), batch_size=1)
    pred_mean = probs.mean(axis=0)
    pred_std = probs.std(axis=0)
    entropy = scipy.stats.entropy(pred_mean)
    _, count = scipy.stats.mode(np.argmax(probs, axis=1))
    variation_ration = 1.0 - count[0]/len(probs)

    test_pred_mean[y_test[i]].append(pred_mean[1])
    test_pred_std[y_test[i]].append(pred_std.mean())
    test_entropy_bayesian[y_test[i]].append(entropy)
    test_variation_ratio[y_test[i]].append(variation_ration)

# Anomaly detection
# by classical prediction entropy
def anomaly_detection(anomaly_score_dict, name):
    threshold = np.logspace(-10.0, 1.0, 1000)
    acc = {}
    for t in threshold:
        tp = 0.0
        tn = 0.0
        for l in anomaly_score_dict:
            if l in train_labels:
                tp += (np.array(anomaly_score_dict[l]) < t).mean()
            else:
                tn += (np.array(anomaly_score_dict[l]) >= t).mean()
        tp /= len(train_labels)
        tn /= 10.0 - len(train_labels)
        bal_acc = (tp + tn)/2.0
        f1_score = 2.0*tp/(2.0 + tp - tn)
        acc[t] = [bal_acc, f1_score, tp, tn]

    print("{}\tscore\tthreshold\tTP\tTN".format(name))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][0], reverse = True)
    print("\tbalanced acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
    print("\tf1 score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))

print('\n\n', '#'*10, network, '#'*10)
anomaly_detection(test_pred_std, "Standard deviation")
anomaly_detection(test_entropy_bayesian, "Entropy")
anomaly_detection(test_variation_ratio, "Variation ratio")

"""

########## batch bayesian ##########

Standard deviation      score   threshold       TP      TN
        balanced acc    0.927   0.207           0.964   0.889
        f1 score        0.929   0.207           0.964   0.889

Entropy                 score   threshold       TP      TN
        balanced acc    0.927   0.187           0.964   0.889
        f1 score        0.929   0.187           0.964   0.889

Variation ratio         score   threshold       TP      TN
        balanced acc    0.927   0.043           0.964   0.889
        f1 score        0.929   0.043           0.964   0.889


########## mlp dropout ##########

Standard deviation  score   threshold   TP      TN
    balanced acc    0.991   0.000       1.000   0.982
    f1 score        0.991   0.000       1.000   0.982

Entropy             score   threshold   TP      TN
    balanced acc    0.991   0.000       1.000   0.982
    f1 score        0.991   0.000       1.000   0.982

Variation ratio     score   threshold   TP      TN
    balanced acc    0.638   0.014       1.000   0.276
    f1 score        0.734   0.014       1.000   0.276

"""
