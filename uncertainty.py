import os
import scipy.stats
import numpy as np
import dataloader
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from layers import Bayesian, PoorBayesian
from layers import ProbabilisticDropout, PoorBayesianConvolution2D
from objectives import bayesian_loss
import time
import pandas as pd
from sklearn import metrics
import tqdm
import pdb

def create_model(network_model, batch_size, input_shape, nb_classes, nb_batchs):
    mean_prior = 0.0
    std_prior = 0.05
    model = Sequential()

    if network_model == 'mlp-bayesian':
        model.add(Bayesian(512, mean_prior, std_prior, batch_input_shape=[batch_size] + input_shape))
        model.add(Activation('relu'))
        model.add(Bayesian(512, mean_prior, std_prior))
        model.add(Activation('relu'))
        model.add(Bayesian(nb_classes, mean_prior, std_prior))
        model.add(Activation('softmax'))
        loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)

    elif network_model == 'mlp-poor-bayesian':
        model.add(PoorBayesian(512, mean_prior, std_prior, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(PoorBayesian(512, mean_prior, std_prior))
        model.add(Activation('relu'))
        model.add(PoorBayesian(nb_classes, mean_prior, std_prior))
        model.add(Activation('softmax'))
        loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)

    elif network_model == 'mlp-dropout':
        model.add(Dense(512, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(ProbabilisticDropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(ProbabilisticDropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'

    elif network_model == 'mlp':
        model.add(Dense(512, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'

    elif network_model == 'convolutional-poor-bayesian':
        model.add(PoorBayesianConvolution2D(mean_prior, std_prior, 32, 3, 3, border_mode='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(PoorBayesianConvolution2D(mean_prior, std_prior, 32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(PoorBayesianConvolution2D(mean_prior, std_prior, 64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(PoorBayesianConvolution2D(mean_prior, std_prior, 64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(PoorBayesian(512, mean_prior, std_prior))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(PoorBayesian(nb_classes, mean_prior, std_prior))
        model.add(Activation('softmax'))
        loss = bayesian_loss(model, mean_prior, std_prior, batch_size, nb_batchs)

    elif network_model == 'convolutional-dropout':
        model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ProbabilisticDropout(0.25))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ProbabilisticDropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(ProbabilisticDropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'

    elif network_model == 'convolutional':
        model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        loss = 'categorical_crossentropy'

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model


def anomaly(experiment_name, network_model, dataset,
            inside_labels, unknown_labels, with_unknown,
            batch_size=100, nb_epochs=100, save_weights=False):

    print('#'*50)
    print('Experiment:', experiment_name)
    print('model:', network_model)
    print('dataset:', dataset)
    print('inside_labels:', str(inside_labels))
    print('unknown_labels:', str(unknown_labels))
    print('batch_size:', batch_size)
    print('nb_epochs:', nb_epochs)
    print('-'*50)

    inside_labels.sort()
    unknown_labels.sort()

    (X_train, y_train), (X_test, y_test) = dataloader.load(dataset,
                                                           inside_labels,
                                                           unknown_labels,
                                                           with_unknown)

    if 'mlp' in network_model:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    input_shape = X_train.shape[1:]
    nb_classes = y_train.shape[1]
    nb_batchs = X_train.shape[0]//batch_size

    model = create_model(network_model, batch_size, input_shape, nb_classes, nb_batchs)

    if network_model.endswith('bayesian') and 'poor' not in network_model:
        mod = X_train.shape[0]%batch_size
        if mod:
            X_train = X_train[:-mod]
            y_train = y_train[:-mod]

    start_time = time.time()
    model.fit(X_train, y_train, nb_epoch=1, batch_size=batch_size)
    end_time = time.time()

    if save_weights:
        path = '{0}_results/weights/{1}_{2}/'
        wu = 'with' if with_unknown else 'without'
        path = path.format(dataset, network_model, wu)
        os.makedirs(path, exist_ok=True)
        model.save_weights(os.path.join(path, experiment_name+'.h5'), overwrite=True)


    measures = {
        'pred_std_mean': x:[] for x in range(10),
        'mean_entropy': x:[] for x in range(10),
        'entropy_mean_class': x:[] for x in range(10),
        'entropy_mean_samples': x:[] for x in range(10),
        'entropy_std_class': x:[] for x in range(10),
        'entropy_std_samples': x:[] for x in range(10),
    }
    test_pred_std = {x:[] for x in range(10)}
    test_entropy = {x:[] for x in range(10)}
    cnt_in = 0
    acc_in = 0

    pbar = tqdm.tqdm(total=X_test.shape[0])
    for i, (x, y) in enumerate(zip(X_test, y_test)):
        if 'poor-bayesian' in network_model:
            probs = model.predict(np.array([x]*batch_size), batch_size=1)
        else:
            probs = model.predict(np.array([x]*batch_size), batch_size=batch_size)

        pred_mean = probs.mean(axis=0)
        pred_std_mean = probs.std(axis=0).mean()
        mean_entropy = scipy.stats.entropy(pred_mean)
        entropy_class = scipy.stats.entropy(probs)
        entropy_samples = scipy.stats.entropy(probs.T)
        entropy_mean_class = entropy_class.mean()
        entropy_mean_samples = entropy_samples.mean()
        entropy_std_class = entropy_class.std()
        entropy_std_samples = entropy_samples.std()
        pdb.set_trace()

        measures['pred_std_mean'][y].append(pred_mean)
        measures['mean_entropy'][y].append(mean_entropy)
        measures['entropy_mean_class'][y].append(entropy_mean_class)
        measures['entropy_mean_samples'][y].append(entropy_mean_samples)
        measures['entropy_std_class'][y].append(entropy_std_class)
        measures['entropy_std_samples'][y].append(entropy_std_samples)

        if y in inside_labels:
            o = np.argmax(pred_mean)
            cnt_in += 1
            if o == inside_labels.index(y):
                acc_in += 1

        pbar.update(1)
    pbar.close()

    # Anomaly detection
    # by classical prediction entropy
    def anomaly_detection(anomaly_score_dict, metric_name, df):
        threshold = np.logspace(-10.0, 1.0, 1000)
        acc = {}
        for t in threshold:
            tp = 0.0
            tn = 0.0
            for l in anomaly_score_dict:
                if l in unknown_labels:
                    continue

                if l in inside_labels:
                    tp += (np.array(anomaly_score_dict[l]) < t).mean()
                else:
                    tn += (np.array(anomaly_score_dict[l]) >= t).mean()

            tp /= len(inside_labels)
            tn /= (10.0 - len(unknown_labels)) - len(inside_labels)
            bal_acc = (tp + tn)/2.0
            f1_score = 2.0*tp/(2.0 + tp - tn)
            acc[t] = [bal_acc, f1_score, tp, tn]

        trues = []
        scores = []
        for l in anomaly_score_dict:
            if l in unknown_labels: continue

            scores += anomaly_score_dict[l]
            if l in inside_labels:
                trues += [0]*len(anomaly_score_dict[l])
            else:
                trues += [1]*len(anomaly_score_dict[l])
        assert len(trues) == len(scores)

        auc = metrics.roc_auc_score(trues, scores)

        print("{}\tscore\tthreshold\tTP\tTN".format(metric_name))
        sorted_acc = sorted(acc.items(), key=lambda x: x[1][0], reverse=True)
        print("\tbalanced_acc\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][0], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        df.set_value(experiment_name, metric_name + "_bal_acc", sorted_acc[0][1][0])

        sorted_acc = sorted(acc.items(), key=lambda x: x[1][1], reverse=True)
        print("\tf1_score\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}".format(sorted_acc[0][1][1], sorted_acc[0][0], sorted_acc[0][1][2], sorted_acc[0][1][3]))
        df.set_value(experiment_name, metric_name + '_f1_score', sorted_acc[0][1][1])
        df.set_value(experiment_name, metric_name + "_auc", auc)
        return df

    print('-'*50)
    df = pd.DataFrame()
    df.set_value(experiment_name, "experiment_name", experiment_name)
    df.set_value(experiment_name, "train_time", end_time - start_time)
    df.set_value(experiment_name, "dataset", dataset)
    df.set_value(experiment_name, "test_acc", acc_in/cnt_in)
    df.set_value(experiment_name, "inside_labels", str(inside_labels))
    df.set_value(experiment_name, "unknown_labels", str(unknown_labels))
    df.set_value(experiment_name, "epochs", nb_epochs)
    df = anomaly_detection(test_pred_std, "pred_std_", df)
    df = anomaly_detection(test_entropy, "entropy_", df)
    return df
