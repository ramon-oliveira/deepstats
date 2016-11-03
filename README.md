# Known Unknowns: Uncertainty Quality in Bayesian Neural Networks

This repository holds the code for the paper "Known Unknowns: Uncertainty Quality in Bayesian Neural Networks" submited to Bayesian Deep Learning NIPS Workshop.

Here we analyse the quality of uncertainty information provided by different Bayesian neural networks when contrasted standard deep learning models. We also propose a novel Bayesian approach for neural networks, similar to the variational approximation of Blundell et al., but much cheaper. We sample the weights only once per training mini-batch, leading to the same expected gradient, and trading off higher variance for computational efficiency (about 10 times faster with a mini-batch of 100). We call that approach One-Sample Bayesian Approximation (OSBA), and investigate whether it achieves better quality of uncertainty information than traditional maximum likelihood models. Techniques like Bayesian-Dropout and OSBA indeed provide better uncertainty information in a well-controlled dataset and deserve further investigation in more contexts. A challenge seems to be an adequate measure of uncertainty for uncontrolled datasets.

## Reproducing results (Python 3.5)

### Installing dependences
```bash
pip install -r requirements.txt
```

### Running an experiment

```bash
python run_experiment.py --dataset=mnist --model=mlp-dropout
```

Available dataset options:
* mnist
* cifar10
* svhn

Available model options:
* mlp
* mlp-dropout
* mlp-poor-bayesian
* mlp-bayesian
* convolutional
* convolutional-dropout
* convolutional-poor-bayesian


### Plotting results

```bash
python plots_anova.py --dataset=mnist
```

## ANOVA Results

### MNIST

![Effects](mnist_results/images/effects.png "Effects")

<img src="mnist_results/images/diff_iou_io.png" alt="Dropout - ML" width="24%"/>
<img src="mnist_results/images/diff_drop_ml.png" alt="Dropout - ML" width="24%"/>
<img src="mnist_results/images/diff_os_ml.png" alt="Dropout - ML" width="24%"/>
<img src="mnist_results/images/diff_os_drop.png" alt="Dropout - ML" width="24%"/>

### CIFAR10 (Updated Results)

![Effects](cifar10_results/images/effects.png "Effects")

<img src="cifar10_results/images/diff_iou_io.png" alt="Dropout - ML" width="24%"/>
<img src="cifar10_results/images/diff_drop_ml.png" alt="Dropout - ML" width="24%"/>
<img src="cifar10_results/images/diff_os_ml.png" alt="Dropout - ML" width="24%"/>
<img src="cifar10_results/images/diff_os_drop.png" alt="Dropout - ML" width="24%"/>
