import numpy as np
import scipy
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

one_way_code = '''
data {
    int K; // Number of groups
    int N; // Number of examples per group
    real y[N, K]; // Observations 
}
parameters {
    real mu; // Mean
    vector[K-1] theta_free; // Effects
    real<lower=0> sigma_likelihood; // Noise std
    real<lower=0> sigma_theta; // Effect std
}
transformed parameters {
  vector[K] theta; // Effects with sum to zero constraint

  for(k in 1:(K-1)) theta[k] <- theta_free[k];
  theta[K] <- -sum(theta_free);
}
model {
    mu ~ normal(0, 10);
    sigma_likelihood ~ uniform(0, 100); // Weak noise std prior
    theta_free ~ normal(0, sigma_theta); // Weak effect prior
    sigma_theta ~ cauchy(0, 25); 
    for (i in 1:N)
        for (j in 1:K)
            y[i][j] ~ normal(mu + theta[j], sigma_likelihood);
}
'''

two_way_code = '''
data {
    int K; // Number of groups
    int N; // Number of examples per group
    real y_in[N, K]; // Observations (inside only)
    real y_out[N, K]; // Observations (including outsiders)
}
parameters {
    real mu_in; // Mean (inside only)
    real mu_out; // Mean (including outsiders)
    vector[K-1] theta_free; // Effects
    real<lower=0> sigma_likelihood; // Noise std
    real<lower=0> sigma_theta; // Effect std
}
transformed parameters {
  vector[K] theta; // Effects with sum to zero constraint

  for(k in 1:(K-1)) theta[k] <- theta_free[k];
  theta[K] <- -sum(theta_free);
}
model {
    sigma_theta ~ cauchy(0, 25);
    theta_free ~ normal(0, sigma_theta); // Weak effect prior

    mu_in ~ normal(0, 100); // Weak mean prior
    mu_out ~ normal(0, 100); // Weak mean prior

    sigma_likelihood ~ uniform(0, 100); // Weak noise std prior (half-normal)
    
    for (i in 1:N)
        for (j in 1:K) {
            // Normal likelihood
            y_in[i][j] ~ normal(mu_in + theta[j], sigma_likelihood); 
            y_out[i][j] ~ normal(mu_out + theta[j], sigma_likelihood);
        }
}
'''

def show_results(fit):
    print(fit)
    fit.plot()
    
def plot_traces(traces, names):
    N = len(traces)+1

    plt.figure()
    for (t, n) in zip(traces, names):
        plt.hist(t, bins = 100, label = n)
    plt.legend()
    plt.xlabel("Logit")
    plt.show()
    
    fig = plt.figure(figsize=(4*N, 4))
    for (t, n, i) in zip(traces, names, range(1, N)):
        ax = fig.add_subplot(1, N, i)   
        pm.plot_posterior(t, varnames=[n], color='#87ceeb', ax = ax)
        ax.set_title(n)
    plt.show()

def effect_difference(effect1, effect2, name1, name2, CI = 95.0):
    diff = effect1 - effect2
    label = str(name1) + ' - ' + str(name2)
    plt.figure(figsize=(4,4))
    plt.hist(diff, bins = 100, label = label)
    plt.legend()
    plt.xlabel("Logit difference")
    plt.show()
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    pm.plot_posterior(effect1-effect2, varnames=[name1, name2], ref_val = 0, color='#87ceeb', ax = ax)
    ax.set_title(label)
    plt.show()
    low_p = (100.0 - CI)/2.0
    high_p = low_p + CI
    print(label, str(CI) + ' CI:', np.percentile(diff, low_p), np.percentile(diff, high_p), 'Pr > 0:', (diff > 0).mean())