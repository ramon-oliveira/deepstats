from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import random

in_dim = 784
out_dim = 2
hidden = 512

batch_size = 1

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
y_train = np.delete(y_train, indexes29, axis=0)

y_train = np_utils.to_categorical(y_train, out_dim)

X_batch = tf.placeholder(tf.float32, [None, in_dim], name='X_batch')
y_batch = tf.placeholder(tf.float32, [None, out_dim], name='y_batch')

## LAYER 1
initial_m0 = tf.zeros([in_dim, hidden], dtype=tf.float32)
mean0 = tf.Variable(initial_m0, name='mean0')
initial_s0 = tf.ones([in_dim, hidden], dtype=tf.float32)
stddev0 = tf.Variable(initial_s0, name='stddev0')
initial_b0 = tf.constant(0.1, shape=[hidden])
b0 = tf.Variable(initial_b0, dtype=tf.float32, name='b0')
W0 = tf.random_normal([in_dim, hidden], dtype=tf.float32, name='W0')

## OUTPUT LAYER
initial_m1 = tf.zeros([hidden, out_dim], dtype=tf.float32)
mean1 = tf.Variable(initial_m1, name='mean1')
initial_s1 = tf.ones([hidden, out_dim], dtype=tf.float32)
stddev1 = tf.Variable(initial_s1, name='stddev1')
initial_b1 = tf.constant(0.1, shape=[out_dim])
b1 = tf.Variable(initial_b1, dtype=tf.float32, name='b1')
W1 = tf.random_normal([hidden, out_dim], dtype=tf.float32, name='W1')

# * is elementwise operator
ff0 = tf.nn.relu(tf.matmul(X_batch, W0*stddev0 + mean0) + b0)
ff1 = tf.matmul(ff0, W1*stddev1 + mean1) + b1
y_out = tf.nn.sigmoid(ff1)
y_out_soft = tf.nn.softmax(y_out)
#cost = tf.nn.sigmoid_cross_entropy_with_logits(ff1, y_batch)
cost = tf.nn.softmax_cross_entropy_with_logits(y_out, y_batch)

total_cost = tf.reduce_mean(cost)

KL0 = (mean0**2)/2.0 + (stddev0**2 - 1.0 - tf.log(stddev0**2))/2.0
KL1 = (mean1**2)/2.0 + (stddev1**2 - 1.0 - tf.log(stddev1**2))/2.0

KL = tf.reduce_sum(KL0) + tf.reduce_sum(KL1)

loss = KL + total_cost

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

def accuracy_train():
    trues = np.argmax(y_train, 1)
    preds = sess.run(tf.arg_max(y_out_soft, 1), feed_dict={X_batch: X_train})
    sm = sum(t == p for t, p in zip(trues, preds))
    return sm/len(y_train)

for epoch in range(1):
    idxs = list(range(len(X_train)))
    random.shuffle(idxs)
    for i, idx in enumerate(idxs):
        if i % 1000 == 0:
            c, k, l = sess.run([cost, KL, loss], 
                               feed_dict={X_batch: X_train, 
                                          y_batch: y_train})
            print('####### Step', i, '########')
            print('cost:', c)
            print('KL:', k)
            print('loss:', l)
            print('accuracy train:', accuracy_train())
            print('-'*30)
            
        sess.run(train, feed_dict={X_batch: np.array([X_train[idx]]), 
                                   y_batch: np.array([y_train[idx]])})

#sess = tf.InteractiveSession()
#sess.close()