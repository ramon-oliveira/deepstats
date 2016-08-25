# Classification uncertainty using Bayesian neural networks

We are using Bayesian neural networks for class anomaly detection. For example, say you have a network that classifies cats and dogs. If you put the image of an ostrich as input, shouldn't the network give some kind of hint that it doesn't understand ostriches? We are testing whether Bayesian neural networks can accomplish that via uncertainty information.

We are trying out two Bayesian approaches:
* Yarin Gal's dropout approximation - see [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://arxiv.org/abs/1506.02142)
* Variational approximation using normal posteriors - see [Weight Uncertainties in Neural Networks](http://arxiv.org/abs/1505.05424)

In the future we may work with HMC approaches for a complete Bayesian inference.

The class anomalies are detected using (marginalized) entropy and prediction variance thresholds. The thresholds are found in a supervised manner.

## Dependencies
* Keras
* Scientific stack: Pandas, Numpy, Scipy, Scikit-learn

## MNIST

#### Standard deviation
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B| 0.927 | 0.929|
| Bayesian NN  | **0.937** | **0.937** |
| MLP dropout  | 0.921 | 0.923 |

#### Entropy
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B| 0.927 | 0.929|
| Bayesian NN  | **0.937** | **0.937** |
| MLP dropout  | 0.922 | 0.924 |
| MLP deterministic  | 0.906 | 0.908 |

#### Variation ratio
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B| 0.927 | 0.929|
| Bayesian NN  | **0.936** | **0.936** |
| MLP dropout  | 0.671 | 0.752 |


## CIFAR10

#### Standard deviation
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B|  |  |
| Bayesian NN  |  |  |
| MLP dropout  | 0.647 | 0.667 |

#### Entropy
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B|  |  |
| Bayesian NN  |  |  |
| MLP dropout  | **0.661** | **0.670** |
| MLP deterministic  | 0.651 | 0.668 |

#### Variation ratio
|  Model       | Balanced accuracy | F1-score |
|--------------|--------------|--------------|
| Bayesian NN B|  |  |
| Bayesian NN  |  |  |
| MLP dropout  | 0.604 | 0.667 |
