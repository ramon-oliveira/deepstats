import argparse
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import pystan
import scipy
import sys
import bayesian_anova


parser = argparse.ArgumentParser(description='ANOVA plot generator')

parser.add_argument('--dataset', dest='dataset', action='store',
                    choices=['mnist','svhn','cifar10'],
                    help='Dataset', required=True)

args = parser.parse_args(sys.argv[1:])
dataset = args.dataset

if dataset == 'mnist':
    nn_model = 'mlp'
    experiments = ['mlp', 'mlp-dropout', 'mlp-poor-bayesian']#, 'mlp-bayesian']
else:
    nn_model = 'convolutional'
    experiments = ['convolutional', 'convolutional-dropout', 'convolutional-poor-bayesian']

cols = ['experiment_name',
        'test_acc',
        'train_time',
        'entropy__auc',
        'entropy_expectation__auc',
        'classifier__auc']

dfs = []
for exp in experiments:
    df_with = pd.read_csv(dataset+'_results/'+exp+'_with_unknown.csv')
    df_with_results = df_with[cols].set_index('experiment_name')
    df_without = pd.read_csv(dataset+'_results/'+exp+'_out_unknown.csv')
    df_without_results = df_without[cols].set_index('experiment_name')
    dfs.append([exp, df_with_results, df_without_results])

results_cols = ['experiment',
                'in_test_acc', 'out_test_acc',
                'in_train_time', 'out_train_time',
                'in_entropy_auc', 'out_entropy_auc',
                'in_entropy_expectation_auc', 'out_entropy_expectation_auc',
                'in_classifier_auc', 'out_classifier_auc']

dfs_results = []
for exp, df_with, df_without in dfs:
    results = pd.DataFrame(columns=results_cols)
    for (in_key, *in_row), (out_key, *out_row) in zip(df_with.itertuples(), df_without.itertuples()):
        assert in_key == out_key
        results.loc[len(results)] = [
            str(in_key),
            in_row[0], out_row[0],
            in_row[1], out_row[1],
            in_row[2], out_row[2],
            in_row[3], out_row[3],
            in_row[4], out_row[4],
        ]
    dfs_results.append([exp, results])

final_results_cols = ['experiment']
for c in results_cols[1:]:
    for exp, _ in dfs_results:
        final_results_cols.append(exp+'_'+c)

final_results = pd.DataFrame(columns=final_results_cols)
for key_row in zip(*[df.itertuples() for exp, df in dfs_results]):
    for i in range(1, len(key_row)):
        _, *prev_row = key_row[i-1]
        _, *row = key_row[i]
        assert prev_row[0] == row[0]

    _, *row = key_row[0]
    new_row = [row[0]]

    for c in range(1, len(results_cols)):
        for _, *row in key_row:
            new_row.append(row[c])

    final_results.loc[len(final_results)] = new_row

model = pystan.StanModel(model_code=bayesian_anova.one_way_code)

out_acc = [nn_model+'_out_classifier_auc',
           nn_model+'-dropout_out_classifier_auc',
           nn_model+'-poor-bayesian_out_classifier_auc']
y_out = final_results[out_acc].values
y_out = scipy.special.logit(y_out)

in_acc = [nn_model+'_in_classifier_auc',
          nn_model+'-dropout_in_classifier_auc',
          nn_model+'-poor-bayesian_in_classifier_auc']
y_in = final_results[in_acc].values
y_in = scipy.special.logit(y_in)


model = pystan.StanModel(model_code=bayesian_anova.two_way_code)

N, K = y_out.shape
data = {'K': K, 'N': N, 'y_in': y_in, 'y_out': y_out}
fit = model.sampling(data=data, iter=100000, warmup = 10000, chains=4, thin=5)

bayesian_anova.show_results(fit)

trace = fit.extract()
deterministic =  scipy.special.expit(trace['theta'][:,0])
dropout = scipy.special.expit(trace['theta'][:,1])
poor_bayesian = scipy.special.expit(trace['theta'][:,2])
in_mean = scipy.special.expit(trace['mu_in'])
out_mean = scipy.special.expit(trace['mu_out'])

traces = [out_mean, in_mean, deterministic, dropout, poor_bayesian]
traces_name = ['Blind Mean', 'Calibrated Mean', 'ML effect', 'BD effect', 'OSBA effect']
fig_hist, figs_effects = bayesian_anova.plot_traces(traces, traces_name, show=False)

fig_hist.savefig(dataset+'_results/images/hist.png')
for name, fig in figs_effects:
    name = name.lower().replace(' ', '_')
    fig.savefig(dataset+'_results/images/'+name+'.png')

fig_diff_drop_ml = bayesian_anova.effect_difference(dropout, deterministic, 'BD', 'ML', show=False)
fig_diff_os_ml = bayesian_anova.effect_difference(poor_bayesian, deterministic, 'OSBA', 'ML', show=False)
fig_diff_os_drop = bayesian_anova.effect_difference(poor_bayesian, dropout, 'OSBA', 'BD', show=False)
fig_diff_iou_io = bayesian_anova.effect_difference(in_mean, out_mean, 'Calibrated', 'Blind', show=False)

fig_diff_drop_ml.savefig(dataset+'_results/images/diff_drop_ml.png')
fig_diff_os_ml.savefig(dataset+'_results/images/diff_os_ml.png')
fig_diff_os_drop.savefig(dataset+'_results/images/diff_os_drop.png')
fig_diff_iou_io.savefig(dataset+'_results/images/diff_iou_io.png')
