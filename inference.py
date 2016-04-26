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
batch_size = 8

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

indexes29 = [i for i, c in enumerate(y_train) if c >= out_dim]
indexes01 = [i for i, c in enumerate(y_train) if c < out_dim]
X_train = np.delete(X_train, indexes29, axis=0)
X_train = X_train.reshape(len(X_train), in_dim, 1)
X_test = X_test.reshape(len(X_test), in_dim, 1)
y_train = np.delete(y_train, indexes29, axis=0)
y_train = np_utils.to_categorical(y_train, out_dim)

nb_batchs = len(X_train)//batch_size
mod = len(X_train) % batch_size

X_batch = tf.placeholder(tf.float32, [batch_size, in_dim, 1], name='X_batch')
y_batch = tf.placeholder(tf.float32, [batch_size, out_dim, 1], name='y_batch')

v = -np.sqrt(6.0 / (in_dim + out_dim))
## LAYER 1
shape = [batch_size, hidden, in_dim]
initial_m0 = tf.random_uniform(shape, -v, v, dtype=tf.float32)
mean0 = tf.Variable(initial_m0, name='mean0')
initial_s0 = tf.random_uniform(shape, -v, v, dtype=tf.float32)
stddev0 = tf.Variable(initial_s0, name='stddev0')
W0 = tf.random_normal(shape, dtype=tf.float32, name='W0')

initial_b0 = tf.random_uniform([hidden, 1], -0.1, 0.1)
b0 = tf.Variable(initial_b0, dtype=tf.float32, name='b0')


## LAYER 2 (OUTPUT)
shape = [batch_size, out_dim, hidden]
initial_m1 = tf.random_uniform(shape, -v, v, dtype=tf.float32)
mean1 = tf.Variable(initial_m1, name='mean1')
initial_s1 = tf.random_uniform(shape, -v, v, dtype=tf.float32)
stddev1 = tf.Variable(initial_s1, name='stddev1')
W1 = tf.random_normal(shape, dtype=tf.float32, name='W1')

initial_b1 = tf.random_uniform([out_dim, 1], -0.1, 0.1)
b1 = tf.Variable(initial_b1, dtype=tf.float32, name='b1')


# * is elementwise operator
ff0 = tf.nn.relu(tf.batch_matmul(W0*tf.exp(stddev0) + mean0, X_batch) + b0)
ff1 = tf.batch_matmul(W1*tf.exp(stddev1) + mean1, ff0) + b1
y_out = ff1
#y_out = tf.nn.relu(ff1)
#y_out = tf.nn.sigmoid(ff1)
y_out = tf.reshape(y_out, [batch_size, out_dim])
y_out_soft = tf.nn.softmax(y_out)
y_batch = tf.reshape(y_batch, [batch_size, out_dim])
cost = tf.nn.softmax_cross_entropy_with_logits(y_out,y_batch)

total_cost = tf.reduce_mean(cost)

KL0 = (mean0**2)/2.0 + (tf.exp(2*stddev0) - 1.0 - 2*stddev0)/2.0
KL1 = (mean1**2)/2.0 + (tf.exp(2*stddev1) - 1.0 - 2*stddev1)/2.0
KL = (tf.reduce_sum(KL0) + tf.reduce_sum(KL1))/nb_batchs

loss = total_cost + KL

vl = [mean0, stddev0, b0, mean1, stddev1, b1]
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


def stats_class(c):
    ids_c = [i for i, t in enumerate(y_test) if t == c]
    X = X_test[ids_c]
    preds_std = []
    for x in X:
        xin = np.array([x]*batch_size)
        outs = sess.run(y_out_soft,feed_dict={X_batch: xin})
        #pred_mean = outs.mean(axis=0)
        pred_std = outs.std(axis=0)
        preds_std.append(pred_std[0])
    preds_std = np.array(preds_std)
    print('Class', c)
    print(' '*4, 'mean std:', preds_std.mean())        


def accuracy_train():
    trues = np.argmax(y_train, 1)
    preds = []
    for i in range(0, len(X_train)-mod, batch_size):
        preds.append(sess.run(tf.arg_max(y_out_soft, 1), 
                              feed_dict={X_batch: X_train[i:i+batch_size]}))
    preds = np.concatenate(preds)
    sm = sum(t == p for t, p in zip(trues, preds))
    return sm/len(y_train)

    
def accuracy_test():
    indexes29 = [i for i, c in enumerate(y_test) if c >= out_dim]
    
    X_test_01 = np.delete(X_test, indexes29, axis=0)
    X_test_01 = X_test_01.reshape(len(X_test_01), in_dim, 1)
    y_test_01 = np.delete(y_test, indexes29, axis=0)
    y_test_01 = np_utils.to_categorical(y_test_01, out_dim)    
    
    trues = np.argmax(y_test_01, 1)
    preds = []
    mod = len(X_test_01) % batch_size
    for i in range(0, len(X_test_01)-mod, batch_size):
        preds.append(sess.run(tf.arg_max(y_out_soft, 1), 
                              feed_dict={X_batch: X_test_01[i:i+batch_size]}))
    preds = np.concatenate(preds)
    sm = sum(t == p for t, p in zip(trues, preds))
    return sm/len(y_test_01)

#saver = tf.train.Saver()
for epoch in range(5):
    print('Epoch:', epoch+1)
    for i in range(0, len(X_train)-mod, batch_size):
        sess.run(train, feed_dict={X_batch: X_train[i:i+batch_size], 
                                   y_batch: y_train[i:i+batch_size]})
        l, tl, KL = sess.run([loss, total_cost, KL], feed_dict={X_batch: X_train[i:i+batch_size], 
                                                    y_batch: y_train[i:i+batch_size]}) 
                                                    
        print('loss:', l, 'KL:', KL, 'total:', tl)
    #saver.save(sess, 'checkpoint.ckpt') #-epoch-'+str(epoch+1)+'.ckpt')
    #print('\t* accuracy train:', accuracy_train())
    print('\t* accuracy test:', accuracy_test())


# Uncertainty prediction
test_pred_mean = {x:[] for x in range(10)}
test_pred_std = {x:[] for x in range(10)}
test_entropy_bayesian_v1 = {x:[] for x in range(10)}
test_entropy_bayesian_v2 = {x:[] for x in range(10)}
test_entropy_deterministic = {x:[] for x in range(10)}
test_variation_ratio = {x:[] for x in range(10)}

for i, x in enumerate(X_test):
    xin = np.array([x]*batch_size)
    outs = sess.run(y_out_soft,feed_dict={X_batch: xin})
    pred_mean = outs.mean(axis=0)[1]
    pred_std = outs.std(axis=0)[1]
    
    test_pred_mean[y_test[i]].append(pred_mean)
    test_pred_std[y_test[i]].append(pred_std)

print(test_pred_std[0][:10]) 
print(test_pred_std[1][:10]) 
print(test_pred_std[2][:10]) 
print(test_pred_std[3][:10]) 
# Anomaly detection
# by classical prediction entropy
inside_labels = [0, 1]
def anomaly_detection(anomaly_score_dict, name):
    threshold = np.linspace(0, 1.0, 10000)
    acc = {}
    for t in threshold:
        tp = 0.0
        tn = 0.0
        for l in anomaly_score_dict:
            if l in inside_labels:
                tp += (np.array(anomaly_score_dict[l]) < t).mean()
            else:
                tn += (np.array(anomaly_score_dict[l]) >= t).mean()
        tp /= len(inside_labels)
        tn /= 10.0 - len(inside_labels)
        bal_acc = (tp + tn)/2.0
        f1_score = 2.0*tp/(2.0 + tp - tn)
        acc[t] = [bal_acc, f1_score, tp, tn]
        
    print("{}\tscore\tthreshold\tTP\tTN".format(name))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][0], reverse = True)
    print("\tbalanced acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
    sorted_acc = sorted(acc.items(), key= lambda x : x[1][1], reverse = True)
    print("\tf1 score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))

anomaly_detection(test_pred_std, "Standard deviation")

sess = tf.InteractiveSession()
sess.close()
