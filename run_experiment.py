import gc
import os
import sys
import argparse
import pandas as pd
from uncertainty import anomaly

parser = argparse.ArgumentParser(description='Model uncertainty experiments.')

parser.add_argument('--dataset', dest='dataset', action='store',
                    choices=['mnist','svhn','cifar10'],
                    help='Dataset', required=True)

parser.add_argument('--model', dest='model', action='store',
                    choices=['mlp','mlp-dropout', 'mlp-poor-bayesian', 'mlp-bayesian',
                             'convolutional', 'convolutional-dropout', 'convolutional-poor-bayesian'],
                    help='Neural Network', required=True)

args = parser.parse_args(sys.argv[1:])

labels = [
    [[0, 1, 4, 8], [7, 9]],
    [[4, 2, 8, 7], [3, 6]],
    [[6, 8, 3, 2], [5, 0]],
    [[5, 3, 7, 8], [2, 4]],
    [[4, 5, 0, 6], [1, 3]],
    [[3, 0, 6, 9], [1, 5]],
    [[9, 6, 1, 8], [4, 3]],
    [[6, 4, 0, 2], [5, 9]],
    [[2, 5, 7, 9], [8, 0]],
    [[1, 8, 6, 2], [4, 9]],
    [[6, 1, 0, 7], [9, 3]],
    [[7, 6, 2, 8], [5, 3]],
    [[6, 5, 7, 1], [8, 4]],
    [[6, 0, 5, 9], [3, 2]],
    [[7, 3, 5, 1], [8, 2]],
    [[8, 7, 3, 0], [5, 6]],
    [[9, 3, 8, 4], [0, 7]],
    [[6, 4, 9, 8], [1, 2]],
    [[8, 2, 3, 7], [0, 9]],
    [[4, 8, 7, 3], [9, 2]],
]

def run_experiment(dataset, model, with_unknown):
    results_folder = dataset+'_results'
    filename = model+'_'+('with' if with_unknown else 'out')+'_unknown.csv'

    try:
        os.mkdir(results_folder)
    except OSError:
        pass

    try:
        df = pd.read_csv(os.path.join(results_folder, filename))
        print('Loaded file:', os.path.join(results_folder, filename))
    except:
        df = pd.DataFrame()

    for idx, (inside_labels, unknown_labels) in enumerate(labels):
        inside_labels.sort()
        unknown_labels.sort()
        aux = df[df.inside_labels == str(inside_labels)]
        aux = aux[aux.unknown_labels == str(unknown_labels)]
        if len(aux) == 3:
            print('Skipping str(inside_labels), str(unknown_labels)')
        for i in range(3 - len(aux)):
            experiment_name = '{}.{}'.format(idx+1, i+1)
            if df.
            out = anomaly(experiment_name, model, dataset,
                          inside_labels, unknown_labels,
                          with_unknown=with_unknown)
            df = df.append(out)
            df.to_csv(os.path.join(results_folder, filename), index=False)
            gc.collect()


run_experiment(args.dataset, args.model, True)
run_experiment(args.dataset, args.model, False)
