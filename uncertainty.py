from keras import backend as K
from keras.engine.topology import Layer
import scipy.stats
import numpy as np
import datasets
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint


class Bayesian(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Bayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        input_dim = input_shape[1]
        shape = [input_shape[0], input_dim, self.output_dim]
        self.W = K.random_normal(shape, mean=0.0, std=std_prior)
        v = np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.log_std = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.log_std, self.bias]

    def call(self, x, mask=None):
        #return K.dot(x, self.W*K.exp(self.log_sigma) + self.mu) + self.bias
        #return K.dot(x, self.W*self.log_sigma + self.mu) + self.bias

        # http://jmlr.org/proceedings/papers/v37/blundell15.pdf
        self.W_sample = self.mean + self.W*K.log(1.0 + K.exp(self.log_std))
        return K.batch_dot(x, self.W_sample) + self.bias

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



batch_size = 5
hidden = 512
dataset = 'cifar10'
# bayesian*, mlp, mlp-deterministic
network = 'bayesian'
train = True
train_labels = [0, 1]
(X_train, y_train), (X_test, y_test) = datasets.load_data(dataset, train_labels)

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
nb_batchs = X_train.shape[0]//batch_size

std_prior = 0.05

def log_gaussian(x, mean, std):
    return -K.log(2*np.pi)/2.0 - K.log(K.abs(std)) - (x-mean)**2/(2*std**2)

def log_gaussian2(x, mean, log_std):
    log_var = 2*log_std
    return -K.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*K.exp(log_var))


def bayesian_loss(y_true, y_pred):
    #log_likelihood = objectives.categorical_crossentropy(y_true, y_pred)
    log_p = K.variable(0.0)
    log_q = K.variable(0.0)
    for layer in model.layers:
        if type(layer) is Bayesian:
            mean = layer.mean
            log_std = layer.log_std
            W_sample = layer.W_sample
            # prior
            log_p += K.sum(log_gaussian(W_sample, 0.0, std_prior))
            # posterior
            log_q += K.sum(log_gaussian2(W_sample, mean, log_std))

    log_likelihood = K.sum(log_gaussian(y_true, y_pred, std_prior))

    return K.sum((log_q - log_p)/nb_batchs - log_likelihood)/batch_size


model = Sequential()
if 'bayesian' in network:
    model.add(Bayesian(hidden, batch_input_shape=[batch_size, in_dim]))
    model.add(Activation('relu'))
    model.add(Bayesian(out_dim))
    model.add(Activation('softmax'))
    loss = bayesian_loss
    # loss = 'categorical_crossentropy' #bayesian_loss
elif network == 'mlp':
    model.add(Dense(hidden, input_shape=[in_dim]))
    model.add(Activation('relu'))
    model.add(MyDropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('softmax'))
    loss = 'categorical_crossentropy'
elif network == 'mlp-deterministic':
    model.add(Dense(hidden, input_shape=[in_dim]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim))
    model.add(Activation('softmax'))
    loss = 'categorical_crossentropy'


model.compile(loss=loss, optimizer='adadelta', metrics=['accuracy'])

#weights_file = 'weights/'+dataset+'/'+network+'-best-weights.h5'
if train:
    cbs = []
    #cbs.append(ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True))
    model.fit(X_train, y_train, nb_epoch=1, batch_size=batch_size, validation_split=0.2, callbacks=cbs)

#model.load_weights(weights_file)
'''
test_pred_mean = {x:[] for x in range(10)}
test_pred_std = {x:[] for x in range(10)}
test_entropy_bayesian = {x:[] for x in range(10)}
test_variation_ratio = {x:[] for x in range(10)}

for i, x in enumerate(X_test):
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
'''


"""
--------------------------- MNIST ----------------------------


 ########## bayesian mega blaster ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.943   0.011       0.940   0.946
    f1 score    0.943   0.011       0.940   0.946
Entropy score   threshold   TP  TN
    balanced acc    0.942   0.057       0.941   0.943
    f1 score    0.942   0.061       0.945   0.939
Variation ratio score   threshold   TP  TN
    balanced acc    0.787   0.014       0.995   0.578
    f1 score    0.823   0.014       0.995   0.578


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


########## bayesian ##########

Standard deviation      score   threshold       TP      TN
    balanced acc        0.937   0.007           0.943   0.930
    f1 score            0.937   0.007           0.943   0.930

Entropy                 score   threshold       TP      TN
    balanced acc        0.937   0.000           0.934   0.940
    f1 score            0.937   0.008           0.943   0.930

Variation ratio         score   threshold       TP      TN
    balanced acc        0.936   0.014           0.944   0.928
    f1 score            0.936   0.014           0.944   0.928


########## mlp dropout ##########

Standard deviation      score   threshold       TP      TN
    balanced acc        0.921   0.000           0.932   0.910
    f1 score            0.923   0.000           0.945   0.897

Entropy                 score   threshold       TP      TN
    balanced acc        0.922   0.002           0.945   0.900
    f1 score            0.924   0.002           0.945   0.900

Variation ratio         score   threshold       TP      TN
    balanced acc        0.671   0.014           0.999   0.344
    f1 score            0.752   0.014           0.999   0.344


########## mlp-deterministic ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.500   0.014       1.000   0.000
    f1 score    0.667   0.014       1.000   0.000
Entropy score   threshold   TP  TN
    balanced acc    0.906   0.000       0.930   0.883
    f1 score    0.908   0.000       0.936   0.875
Variation ratio score   threshold   TP  TN
    balanced acc    0.500   0.014       1.000   0.000
    f1 score    0.667   0.014       1.000   0.000

-------------------- CIFAR10 ----------------------


 ########## bayesian mega blaster ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.578   0.000       0.333   0.823
    f1 score    0.667   3.037       1.000   0.000
Entropy score   threshold   TP  TN
    balanced acc    0.578   0.000       0.336   0.821
    f1 score    0.667   3.037       1.000   0.000
Variation ratio score   threshold   TP  TN
    balanced acc    0.560   0.014       0.532   0.588
    f1 score    0.667   3.037       1.000   0.000

 ########## mlp dropout ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.647   0.042       0.472   0.823
    f1 score    0.667   0.212       0.984   0.035
Entropy score   threshold   TP  TN
    balanced acc    0.661   0.280       0.553   0.769
    f1 score    0.670   0.502       0.761   0.489
Variation ratio score   threshold   TP  TN
    balanced acc    0.604   0.014       0.726   0.482
    f1 score    0.667   0.352       0.956   0.089


########## bayesian-batch ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.503   0.477       0.053   0.953
    f1 score    0.667   3.037       1.000   0.000
Entropy score   threshold   TP  TN
    balanced acc    0.507   0.663       0.096   0.917
    f1 score    0.667   3.037       1.000   0.000
Variation ratio score   threshold   TP  TN
    balanced acc    0.509   0.465       0.703   0.315
    f1 score    0.667   3.037       1.000   0.000


 ########## mlp-deterministic ##########
Standard deviation  score   threshold   TP  TN
    balanced acc    0.500   0.014       1.000   0.000
    f1 score    0.667   0.014       1.000   0.000
Entropy score   threshold   TP  TN
    balanced acc    0.651   0.169       0.588   0.715
    f1 score    0.668   0.489       0.833   0.340
Variation ratio score   threshold   TP  TN
    balanced acc    0.500   0.014       1.000   0.000
    f1 score    0.667   0.014       1.000   0.000

"""
