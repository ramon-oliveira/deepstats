import gc
from uncertainty import anomaly
import pandas as pd

params = [
    ['test_mnist_4labels_2unknown_1.1', 'bayesian', 'mnist', [0, 1, 4, 8], [7, 9]],
    ['test_mnist_4labels_2unknown_1.2', 'bayesian', 'mnist', [0, 1, 4, 8], [7, 9]],
    ['test_mnist_4labels_2unknown_1.3', 'bayesian', 'mnist', [0, 1, 4, 8], [7, 9]],
    ['test_mnist_4labels_2unknown_2.1', 'bayesian', 'mnist', [4, 2, 8, 7], [3, 6]],
    ['test_mnist_4labels_2unknown_2.2', 'bayesian', 'mnist', [4, 2, 8, 7], [3, 6]],
    ['test_mnist_4labels_2unknown_2.3', 'bayesian', 'mnist', [4, 2, 8, 7], [3, 6]],
    ['test_mnist_4labels_2unknown_3.1', 'bayesian', 'mnist', [6, 8, 3, 2], [5, 0]],
    ['test_mnist_4labels_2unknown_3.2', 'bayesian', 'mnist', [6, 8, 3, 2], [5, 0]],
    ['test_mnist_4labels_2unknown_3.3', 'bayesian', 'mnist', [6, 8, 3, 2], [5, 0]],
    ['test_mnist_4labels_2unknown_4.1', 'bayesian', 'mnist', [5, 3, 7, 8], [2, 4]],
    ['test_mnist_4labels_2unknown_4.2', 'bayesian', 'mnist', [5, 3, 7, 8], [2, 4]],
    ['test_mnist_4labels_2unknown_4.3', 'bayesian', 'mnist', [5, 3, 7, 8], [2, 4]],
    ['test_mnist_4labels_2unknown_5.1', 'bayesian', 'mnist', [4, 5, 0, 6], [1, 3]],
    ['test_mnist_4labels_2unknown_5.2', 'bayesian', 'mnist', [4, 5, 0, 6], [1, 3]],
    ['test_mnist_4labels_2unknown_5.3', 'bayesian', 'mnist', [4, 5, 0, 6], [1, 3]],
    ['test_mnist_4labels_2unknown_6.1', 'bayesian', 'mnist', [3, 0, 6, 9], [1, 5]],
    ['test_mnist_4labels_2unknown_6.2', 'bayesian', 'mnist', [3, 0, 6, 9], [1, 5]],
    ['test_mnist_4labels_2unknown_6.3', 'bayesian', 'mnist', [3, 0, 6, 9], [1, 5]],
    ['test_mnist_4labels_2unknown_7.1', 'bayesian', 'mnist', [9, 6, 1, 8], [4, 3]],
    ['test_mnist_4labels_2unknown_7.2', 'bayesian', 'mnist', [9, 6, 1, 8], [4, 3]],
    ['test_mnist_4labels_2unknown_7.3', 'bayesian', 'mnist', [9, 6, 1, 8], [4, 3]],
    ['test_mnist_4labels_2unknown_8.1', 'bayesian', 'mnist', [6, 4, 0, 2], [5, 9]],
    ['test_mnist_4labels_2unknown_8.2', 'bayesian', 'mnist', [6, 4, 0, 2], [5, 9]],
    ['test_mnist_4labels_2unknown_8.3', 'bayesian', 'mnist', [6, 4, 0, 2], [5, 9]],
    ['test_mnist_4labels_2unknown_9.1', 'bayesian', 'mnist', [2, 5, 7, 9], [8, 0]],
    ['test_mnist_4labels_2unknown_9.2', 'bayesian', 'mnist', [2, 5, 7, 9], [8, 0]],
    ['test_mnist_4labels_2unknown_9.3', 'bayesian', 'mnist', [2, 5, 7, 9], [8, 0]],
    ['test_mnist_4labels_2unknown_10.1', 'bayesian', 'mnist', [1, 8, 6, 2], [4, 9]],
    ['test_mnist_4labels_2unknown_10.2', 'bayesian', 'mnist', [1, 8, 6, 2], [4, 9]],
    ['test_mnist_4labels_2unknown_10.3', 'bayesian', 'mnist', [1, 8, 6, 2], [4, 9]],
    ['test_mnist_4labels_2unknown_11.1', 'bayesian', 'mnist', [6, 1, 0, 7], [9, 3]],
    ['test_mnist_4labels_2unknown_11.2', 'bayesian', 'mnist', [6, 1, 0, 7], [9, 3]],
    ['test_mnist_4labels_2unknown_11.3', 'bayesian', 'mnist', [6, 1, 0, 7], [9, 3]],
    ['test_mnist_4labels_2unknown_12.1', 'bayesian', 'mnist', [7, 6, 2, 8], [5, 3]],
    ['test_mnist_4labels_2unknown_12.2', 'bayesian', 'mnist', [7, 6, 2, 8], [5, 3]],
    ['test_mnist_4labels_2unknown_12.3', 'bayesian', 'mnist', [7, 6, 2, 8], [5, 3]],
    ['test_mnist_4labels_2unknown_13.1', 'bayesian', 'mnist', [6, 5, 7, 1], [8, 4]],
    ['test_mnist_4labels_2unknown_13.2', 'bayesian', 'mnist', [6, 5, 7, 1], [8, 4]],
    ['test_mnist_4labels_2unknown_13.3', 'bayesian', 'mnist', [6, 5, 7, 1], [8, 4]],
    ['test_mnist_4labels_2unknown_14.1', 'bayesian', 'mnist', [6, 0, 5, 9], [3, 2]],
    ['test_mnist_4labels_2unknown_14.2', 'bayesian', 'mnist', [6, 0, 5, 9], [3, 2]],
    ['test_mnist_4labels_2unknown_14.3', 'bayesian', 'mnist', [6, 0, 5, 9], [3, 2]],
    ['test_mnist_4labels_2unknown_15.1', 'bayesian', 'mnist', [7, 3, 5, 1], [8, 2]],
    ['test_mnist_4labels_2unknown_15.2', 'bayesian', 'mnist', [7, 3, 5, 1], [8, 2]],
    ['test_mnist_4labels_2unknown_15.3', 'bayesian', 'mnist', [7, 3, 5, 1], [8, 2]],
    ['test_mnist_4labels_2unknown_16.1', 'bayesian', 'mnist', [8, 7, 3, 0], [5, 6]],
    ['test_mnist_4labels_2unknown_16.2', 'bayesian', 'mnist', [8, 7, 3, 0], [5, 6]],
    ['test_mnist_4labels_2unknown_16.3', 'bayesian', 'mnist', [8, 7, 3, 0], [5, 6]],
    ['test_mnist_4labels_2unknown_17.1', 'bayesian', 'mnist', [9, 3, 8, 4], [0, 7]],
    ['test_mnist_4labels_2unknown_17.2', 'bayesian', 'mnist', [9, 3, 8, 4], [0, 7]],
    ['test_mnist_4labels_2unknown_17.3', 'bayesian', 'mnist', [9, 3, 8, 4], [0, 7]],
    ['test_mnist_4labels_2unknown_18.1', 'bayesian', 'mnist', [6, 4, 9, 8], [1, 2]],
    ['test_mnist_4labels_2unknown_18.2', 'bayesian', 'mnist', [6, 4, 9, 8], [1, 2]],
    ['test_mnist_4labels_2unknown_18.3', 'bayesian', 'mnist', [6, 4, 9, 8], [1, 2]],
    ['test_mnist_4labels_2unknown_19.1', 'bayesian', 'mnist', [8, 2, 3, 7], [0, 9]],
    ['test_mnist_4labels_2unknown_19.2', 'bayesian', 'mnist', [8, 2, 3, 7], [0, 9]],
    ['test_mnist_4labels_2unknown_19.3', 'bayesian', 'mnist', [8, 2, 3, 7], [0, 9]],
    ['test_mnist_4labels_2unknown_20.1', 'bayesian', 'mnist', [4, 8, 7, 3], [9, 2]],
    ['test_mnist_4labels_2unknown_20.2', 'bayesian', 'mnist', [4, 8, 7, 3], [9, 2]],
    ['test_mnist_4labels_2unknown_20.3', 'bayesian', 'mnist', [4, 8, 7, 3], [9, 2]],
]


try:
    df = pd.read_csv('bayesian_uncertainty_with_unknown.csv')
except:
    df = pd.DataFrame()

for experiment, network, dataset, inside_labels, unknown_labels in params:
    out = anomaly(experiment, network, dataset, inside_labels, unknown_labels, with_unknown=True)
    df = df.append(out)
    df.to_csv('bayesian_uncertainty_with_unknown.csv', index=False)
    gc.collect()


try:
    df = pd.read_csv('bayesian_uncertainty_without_unknown.csv')
except:
    df = pd.DataFrame()

for experiment, network, dataset, inside_labels, unknown_labels in params:
    out = anomaly(experiment, network, dataset, inside_labels, unknown_labels, with_unknown=False)
    df = df.append(out)
    df.to_csv('bayesian_uncertainty_without_unknown.csv', index=False)
    gc.collect()
