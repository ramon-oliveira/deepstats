from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
import numpy as np

in_dim = 784
out_dim = 1
hidden = 512

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

indexes29 = [i for i, c in enumerate(y_train) if c >= 2]
indexes01 = [i for i, c in enumerate(y_train) if c < 2]
#test_indexes29 = [i for i, c in enumerate(y_test) if c >= 2]
#test_indexes01 = [i for i, c in enumerate(y_test) if c < 2]
X_train = np.delete(X_train, indexes29, axis=0)
y_train = np.delete(y_train, indexes29, axis=0)
y_train = y_train.reshape([len(y_train), 1])
#y_train = np_utils.to_categorical(y_train, out_dim)


X_batch = tf.placeholder(tf.float32, [None, in_dim], name='X_batch')
y_batch = tf.placeholder(tf.float32, [None, out_dim], name='y_batch')

initial_m0 = tf.zeros([in_dim, hidden], dtype=tf.float32)
mean0 = tf.Variable(initial_m0, name='mean0')
initial_s0 = tf.ones([in_dim, hidden], dtype=tf.float32)
stddev0 = tf.Variable(initial_s0, name='stddev0')
initial_b0 = tf.constant(0.1, shape=[hidden])
b0 = tf.Variable(initial_b0, dtype=tf.float32, name='b0')
W0 = tf.random_normal([in_dim, hidden], dtype=tf.float32, name='W0')

initial_m1 = tf.zeros([hidden, out_dim], dtype=tf.float32)
mean1 = tf.Variable(initial_m1, name='mean1')
initial_s1 = tf.ones([hidden, out_dim], dtype=tf.float32)
stddev1 = tf.Variable(initial_s1, name='stddev1')
initial_b1 = tf.constant(0.1, shape=[out_dim])
b1 = tf.Variable(initial_b1, dtype=tf.float32, name='b1')
W1 = tf.random_normal([hidden, out_dim], dtype=tf.float32, name='W1')

# * is elementwise operator
ff0 = tf.nn.relu(tf.matmul(X_batch, tf.mul(W0, stddev0) + mean0) + b0)
ff1 = tf.matmul(ff0, tf.mul(W1, stddev1) + mean1) + b1
y_out = tf.nn.sigmoid(ff1)

cost = tf.nn.sigmoid_cross_entropy_with_logits(ff1, y_batch)
#cost = -y_batch*tf.clip_by_value(tf.log(y_out), -1e6, 1e6) - \
#       (1.0 - y_batch)*tf.clip_by_value(tf.log(1.0 - y_out), -1e6, 1e6) 
#cost = y_out - y_out*y_batch + tf.log(1 + tf.exp(-y_out))
       
#cost = y_batch*-tf.log(y_out) + (1.0 - y_batch)*-tf.log(1.0 - y_out)

KL0 = (mean0**2)/2.0 + (stddev0**2 - 1.0 - tf.log(stddev0**2))/2.0
KL1 = (mean1**2)/2.0 + (stddev1**2 - 1.0 - tf.log(stddev1**2))/2.0

KL = tf.reduce_sum(KL0) + tf.reduce_sum(KL1)

loss = KL + cost

ws = [mean0, stddev0, b0, mean1, stddev1, b1]
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1001):
    if i % 100 == 0:
        o, oi, c, k, l = sess.run([y_out, ff1, cost, KL, loss], 
                                  feed_dict={X_batch: X_train[:1], y_batch: y_train[:1]})
        print('Step', i+1)
        print('output:')
        print(o, oi, y_train[0])
        print('cost:', c)
        print('KL:', k)
        print('loss:', l)
        m0, s0 = sess.run([mean0, stddev0])
        print('mean0 sum:', np.sum(m0) ,'stddev0 sum:', np.sum(s0))
    sess.run(train, feed_dict={X_batch: X_train[i:i+1], 
                               y_batch: y_train[i:i+1]})
    #grad = tf.gradients(loss, [mean0])[0]
    #vg = sess.run(grad, feed_dict={X_batch: X_train[:1], y_batch: y_train[:1]})
    #print('gradient:', len(vg), len(vg[0]))
    #print('gradient:', vg)

#sess = tf.InteractiveSession()
sess.close()