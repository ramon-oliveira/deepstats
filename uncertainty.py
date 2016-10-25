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
import pickle as pkl
from sklearn import linear_model

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


def get_measures(Xs, ys, model, batch_size, inside_labels):
    measures = {
        'pred_std_mean': {x:[] for x in range(10)},
        'mean_entropy': {x:[] for x in range(10)},
        'entropy_mean_class': {x:[] for x in range(10)},
        'entropy_mean_samples': {x:[] for x in range(10)},
        'entropy_std_class': {x:[] for x in range(10)},
        'entropy_std_samples': {x:[] for x in range(10)},
    }
    cnt_in = 0
    acc_in = 0
    pbar = tqdm.tqdm(total=Xs.shape[0])
    for i, (x, y) in enumerate(zip(Xs, ys)):
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

        measures['pred_std_mean'][y].append(pred_std_mean)
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
    return measures, acc_in/cnt_in


def uncertainty_classifier(measures, inside_labels, unknown_labels):
    features_vectors = []
    labels = []
    for l in range(10):
        if l in unknown_labels:
            continue
        n = len(measures['entropy_std_samples'][l])
        for i in range(n):
            f = [
                measures['mean_entropy'][l][i],
                measures['pred_std_mean'][l][i],
                measures['entropy_std_samples'][l][i],
                measures['entropy_mean_samples'][l][i],
            ]
            features_vectors.append(f)
            labels.append(l in inside_labels)

    features_vectors = np.array(features_vectors, dtype=np.float64)
    labels = np.array(labels, dtype=np.bool)

    idxs = np.arange(len(features_vectors))
    np.random.shuffle(idxs)
    idxs_train = idxs[:int(idxs.shape[0]*0.7)]
    idxs_test = idxs[int(idxs.shape[0]*0.7):]

    X_train = features_vectors[idxs_train]
    y_train = labels[idxs_train]
    X_test = features_vectors[idxs_test]
    y_test = labels[idxs_test]

    lr = linear_model.LogisticRegressionCV(Cs=100, scoring='roc_auc').fit(X_train, y_train)
    return lr


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
    model.fit(X_train, y_train, nb_epoch=nb_epochs, batch_size=batch_size)
    end_time = time.time()

    if save_weights:
        path = '{0}_results/weights/{1}_{2}/'
        wu = 'with' if with_unknown else 'without'
        path = path.format(dataset, network_model, wu)
        os.makedirs(path, exist_ok=True)
        model.save_weights(os.path.join(path, experiment_name+'.h5'), overwrite=True)

    bs = batch_size
    if 'poor-bayesian' in network_model:
        bs = 1

    measures_train, train_acc = get_measures(X_train, y_train, model, bs, inside_labels)
    measures_test, test_acc = get_measures(X_test, y_test, model, bs, inside_labels)

    clf = uncertainty_classifier(measures_train, inside_labels, unknown_labels)
    measures_test['classifier'] = {l:[] for l in range(10)}
    for l in range(10):
        n = len(measures_test['entropy_std_samples'][l])
        for i in range(n):
            f = [
                measures_test['mean_entropy'][l][i],
                measures_test['pred_std_mean'][l][i],
                measures_test['entropy_std_samples'][l][i],
                measures_test['entropy_mean_samples'][l][i],
            ]
            p = clf.predict_proba([f])[0, 1]
            measures_test['classifier'][l].append(p)

    pdb.set_trace()
    pkl.dump(measures_train, open('/work/roliveira/cifar10_measures_train_without.pkl', 'wb'))
    pkl.dump(measures_test, open('/work/roliveira/cifar10_measures_test_without.pkl', 'wb'))

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

        sorted_acc = sorted(acc.items(), key=lambda x: x[1][0], reverse=True)
        df.set_value(experiment_name, metric_name + '_bal_acc', sorted_acc[0][1][0])
        bal_acc = sorted_acc[0][1][0]

        sorted_acc = sorted(acc.items(), key=lambda x: x[1][1], reverse=True)
        df.set_value(experiment_name, metric_name + '_f1_score', sorted_acc[0][1][1])
        f1_score = sorted_acc[0][1][1]
        df.set_value(experiment_name, metric_name + '_auc', auc)

        msg = '{0}: (auc, {1:.2f}), (bal_acc, {2:.2f}), (f1_score, {3:.2f})'
        print(msg.format(metric_name, auc, bal_acc, f1_score))

        return df

    print('-'*50)
    df = pd.DataFrame()
    df.set_value(experiment_name, 'experiment_name', experiment_name)
    df.set_value(experiment_name, 'train_time', end_time - start_time)
    df.set_value(experiment_name, 'dataset', dataset)
    df.set_value(experiment_name, 'test_acc', acc_in/cnt_in)
    df.set_value(experiment_name, 'inside_labels', str(inside_labels))
    df.set_value(experiment_name, 'unknown_labels', str(unknown_labels))
    df.set_value(experiment_name, 'epochs', nb_epochs)
    df = anomaly_detection(measures_test['pred_std_mean'], 'pred_std_', df)
    df = anomaly_detection(measures_test['mean_entropy'], 'entropy_', df)
    df = anomaly_detection(measures_test['entropy_mean_samples'], 'entropy_expectation_', df)
    df = anomaly_detection(measures_test['classifier'], 'classifier_', df)

    return df
