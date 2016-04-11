# from keras.datasets import mnist
import tensorflow as tf
import numpy as np

#(X_train, y_train), (X_test, y_test) = mnist.load_data()

in_dim = 1
out_dim = 1
hidden = 1


X_batch = tf.placeholder(tf.float32, [None, in_dim])
y_batch = tf.placeholder(tf.float32, [None, out_dim])

mean0 = [[tf.Variable(0, dtype=tf.float32) for j in range(hidden)] for i in range(in_dim)]
stddev0 = [[tf.Variable(0, dtype=tf.float32) for j in range(hidden)] for i in range(in_dim)]
W0 = tf.Variable([in_dim, hidden], dtype=tf.float32)
b0 = tf.Variable([hidden, 1], dtype=tf.float32)


normals = [[tf.random_normal(0, mean=mean0[i][j], stddev=stddev0[i][j]) for j in range(hidden)] 
           for i in range(in_dim)]

W0.assign(normals)

#W1 = tf.random_normal([512, out_dim], mean=0, stddev=1)
#b1 = tf.random_normal([out_dim, 1], mean=0, stddev=1)

#y_out = tf.nn.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(X_batch, W0) + b0), W1) + b1)

# cost = y_batch*-tf.log(tf.sigmoid(y_out)) + (1 - y_batch)*-tf.log(1 - tf.sigmoid(y_out)) 
#cost = tf.nn.sigmoid_cross_entropy_with_logits(y_out, y_batch)

# KL = 