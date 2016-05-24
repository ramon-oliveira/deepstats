Uncertainty
===========


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
