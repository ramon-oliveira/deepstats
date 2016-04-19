from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import random
np.random.seed(42)
random.seed(42)

in_dim = 784
out_dim = 2
hidden = 512
batch_size = 128

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

indexes29 = [i for i, c in enumerate(y_train) if c >= 2]
indexes01 = [i for i, c in enumerate(y_train) if c < 2]
X_train = np.delete(X_train, indexes29, axis=0)
X_train = X_train.reshape(len(X_train), in_dim, 1)
y_train = np.delete(y_train, indexes29, axis=0)
y_train = np_utils.to_categorical(y_train, out_dim)

nb_batchs = len(X_train)//batch_size
mod = len(X_train) % batch_size

X_batch = tf.placeholder(tf.float32, [batch_size, in_dim, 1], name='X_batch')
y_batch = tf.placeholder(tf.float32, [batch_size, out_dim, 1], name='y_batch')

## LAYER 1
shape = [batch_size, hidden, in_dim]

initial_m0 = tf.zeros(shape, dtype=tf.float32)
mean0 = tf.Variable(initial_m0, name='mean0')
initial_s0 = tf.ones(shape, dtype=tf.float32)
stddev0 = tf.Variable(initial_s0, name='stddev0')
W0 = tf.random_normal(shape, dtype=tf.float32, name='W0')

initial_b0 = tf.random_uniform([hidden, 1], -0.1, 0.1)
b0 = tf.Variable(initial_b0, dtype=tf.float32, name='b0')


## LAYER 2 (OUTPUT)
shape = [batch_size, out_dim, hidden]
initial_m1 = tf.zeros(shape, dtype=tf.float32)
mean1 = tf.Variable(initial_m1, name='mean1')
initial_s1 = tf.ones(shape, dtype=tf.float32)
stddev1 = tf.Variable(initial_s1, name='stddev1')
W1 = tf.random_normal(shape, dtype=tf.float32, name='W1')

initial_b1 = tf.random_uniform([out_dim], -0.1, 0.1)
b1 = tf.Variable(initial_b1, dtype=tf.float32, name='b1')


# * is elementwise operator
ff0 = tf.nn.relu(tf.batch_matmul(W0*tf.exp(stddev0) + mean0, X_batch))
ff1 = tf.batch_matmul(W1*tf.exp(stddev1) + mean1, ff0)# + b1
y_out = tf.nn.relu(ff1)
#y_out = tf.nn.sigmoid(ff1)
y_out = tf.reshape(y_out, [batch_size, out_dim])
y_out_soft = tf.nn.softmax(y_out)
y_batch = tf.reshape(y_batch, [batch_size, out_dim])
cost = tf.nn.softmax_cross_entropy_with_logits(y_out,y_batch)

total_cost = tf.reduce_mean(cost)

KL0 = (mean0**2)/2.0 + (tf.exp(2*stddev0) - 1.0 - 2*stddev0)/2.0
KL1 = (mean1**2)/2.0 + (tf.exp(2*stddev1) - 1.0 - 2*stddev1)/2.0

KL = tf.reduce_sum(KL0) + tf.reduce_sum(KL1)

loss = (KL + total_cost)/nb_batchs
#loss = total_cost

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

def accuracy_train():
    trues = np.argmax(y_train, 1)
    preds = []
    for i in range(0, len(X_train)-mod, batch_size):
        preds.append(sess.run(tf.arg_max(y_out_soft, 1), 
                              feed_dict={X_batch: X_train[i:i+batch_size]}))
    preds = np.concatenate(preds)
    sm = sum(t == p for t, p in zip(trues, preds))
    return sm/len(y_train)

for epoch in range(1):
    idxs = list(range(len(X_train)))
    random.shuffle(idxs)
    #for i, idx in enumerate(idxs):
    for i in range(0, len(X_train), batch_size):
        c, k, l = sess.run([total_cost, KL, loss], 
                           feed_dict={X_batch: X_train[:batch_size], 
                                      y_batch: y_train[:batch_size]})
        print('####### Step', i, '########')
        print('cost:', c)
        print('KL:', k)
        print('loss:', l)
        print('accuracy train:', accuracy_train())
        print('-'*30)
            
        sess.run(train, feed_dict={X_batch: X_train[i:i+batch_size], 
                                   y_batch: y_train[i:i+batch_size]})

#sess = tf.InteractiveSession()
#sess.close()