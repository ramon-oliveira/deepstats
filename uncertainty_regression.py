from keras import backend as K
from keras.engine.topology import Layer
import scipy.stats
import numpy as np
import datasets
from keras.models import Sequential
from keras.layers import Activation, Dense, ELU, Dropout
from keras import objectives
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn import datasets
import matplotlib.pyplot as plt

class Bayesian(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Bayesian, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = [input_dim, self.output_dim]
        self.W = K.random_normal(shape)
        v = 1.0 # np.sqrt(6.0 / (input_dim + self.output_dim))
        self.mean = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.log_stdev = K.variable(np.random.uniform(low=-v, high=v, size=shape))
        self.bias = K.variable(np.random.uniform(low=-v, high=v, size=[self.output_dim]))

        self.trainable_weights = [self.mean, self.bias, self.log_stdev]

    def call(self, x, mask=None):
        return K.batch_dot(x, self.W*K.exp(self.log_stdev) + self.mean) + self.bias
        #return K.dot(x, self.W*self.log_stdev + self.mean) + self.bias

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



batch_size = 32
hidden = 512
dataset = 'diabetes' #mnist
# bayesian-batch, bayesian, mlp, mlp-deterministic
network = 'mlp' #'bayesian'
train = True
train_labels = [0, 1] # automobile, dog
#(X_train, y_train), (X_test, y_test) = datasets.load_data(dataset, train_labels)
reg = l2(l=0.001)

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
X_train = diabetes_X[:-50]
X_test = diabetes_X[-50:]
y_train = diabetes.target[:-50]
y_test = diabetes.target[-50:]


# Standardize
xmu = diabetes.data[:, np.newaxis, 2].mean()
xsigma = diabetes.data[:, np.newaxis, 2].std()

ymu = diabetes.target.mean()
ysigma = diabetes.target.std()

X_train = (X_train - xmu)/xsigma
y_train = (y_train - ymu)/ysigma


X_test = (X_test - xmu)/xsigma
y_test = (y_test - ymu)/ysigma

in_dim = 1
out_dim = 1
nb_batchs = X_train.shape[0]//batch_size

def bayesian_loss(y_true, y_pred):
    ce = objectives.mean_squared_error(y_true, y_pred)
    kl = K.variable(0.0)
    for layer in model.layers:
        if type(layer) is Bayesian:
            mean = layer.mean
            log_stdev = layer.log_stdev

            #DKL_hidden = (1.0 + 2.0*W1_log_var - W1_mu**2.0 - T.exp(2.0*W1_log_var)).sum()/2.0

            kl = kl + K.sum(1.0 + 2.0*log_stdev - mean**2.0 - K.exp(2.0*log_stdev))/2.0
            #kl = kl + K.sum(K.log(1.0/log_stdev) + (log_stdev**2 + mean**2)/2.0 - 0.5)
            #(mean**2)/2 + (K.exp(2*stdev) - 1 - 2*stdev)/2)
    return ce + kl/nb_batchs


model = Sequential()
if 'bayesian' in network:
    model.add(Bayesian(hidden, input_shape=[in_dim]))
    model.add(Activation('relu'))
    model.add(Bayesian(out_dim))
    model.add(Activation('linear'))
    loss = bayesian_loss
    #loss = 'mse' #bayesian_loss
elif network == 'mlp':
    model.add(Dense(hidden, input_shape=[in_dim], W_regularizer=reg))
    model.add(Activation('relu'))
    model.add(MyDropout(0.5))
    model.add(Dense(out_dim, W_regularizer=reg))
    model.add(Activation('linear'))
    loss = 'mse'
elif network == 'mlp-deterministic':
    model.add(Dense(hidden, input_shape=[in_dim], W_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, W_regularizer=reg))
    model.add(Activation('softmax'))
    loss = 'categorical_crossentropy'

#optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-4, nesterov=True)
#optimizer = Adam()
#optimizer = Adagrad()
optimizer = SGD()
#optimizer = Adadelta()#clipnorm=1.0, clipvalue=1.0)
#optimizer = RMSprop()
model.compile(loss=loss, optimizer=optimizer)

weights_file = 'weights/'+dataset+'/'+network+'-best-weights.h5'

#model.load_weights(weights_file)

if train:
    mc = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True)
    model.fit(X_train, y_train, nb_epoch=100, batch_size=batch_size,
              validation_split=0.1, callbacks=[mc])

#model.load_weights(weights_file)
p=0.5
l=1.0
tau = l**2 * (1-p)/(2*X_train.shape[0]*reg.l2)

test_pred_mean = []
test_pred_var = []

for i, x in enumerate(X_test):
    probs = model.predict(np.array([x]*50), batch_size=1)
    pred_mean = probs.mean(axis=0)[0]
    pred_var = probs.var(axis=0)[0]
    
    pred_var += tau**-1

    test_pred_mean.append(pred_mean)
    test_pred_var.append(pred_var)
    
    

plt.plot(y_test, 'o')
t = np.arange(len(X_test))

#plt.errorbar(t, test_pred_mean, yerr=test_pred_std)

upper_bound = np.array(test_pred_mean) + np.array(test_pred_var)
lower_bound = np.array(test_pred_mean) - np.array(test_pred_var)
plt.fill_between(t, lower_bound, upper_bound, facecolor='green', alpha=0.5)
#plt.errorbar(list(range(len(X_test))), test_pred_mean, yerr=test_entropy_bayesian)    

#
## Anomaly detection
## by classical prediction entropy
#def anomaly_detection(anomaly_score_dict, name):
#    threshold = np.logspace(-10.0, 1.0, 1000)
#    acc = {}
#    for t in threshold:
#        tp = 0.0
#        tn = 0.0
#        for l in anomaly_score_dict:
#            if l in train_labels:
#                tp += (np.array(anomaly_score_dict[l]) < t).mean()
#            else:
#                tn += (np.array(anomaly_score_dict[l]) >= t).mean()
#        tp /= len(train_labels)
#        tn /= 10.0 - len(train_labels)
#        bal_acc = (tp + tn)/2.0
#        f1_score = 2.0*tp/(2.0 + tp - tn)
#        acc[t] = [bal_acc, f1_score, tp, tn]
#
#    print("{}\tscore\tthreshold\tTP\tTN".format(name))
#    sorted_acc = sorted(acc.items(), key= lambda x : x[1][0], reverse = True)
#    print("\tbalanced acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
#    sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
#    print("\tf1 score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
#
#print('\n\n', '#'*10, network, '#'*10)
#anomaly_detection(test_pred_std, "Standard deviation")
#anomaly_detection(test_entropy_bayesian, "Entropy")
#anomaly_detection(test_variation_ratio, "Variation ratio")

"""

--------------------------- MNIST ----------------------------

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
