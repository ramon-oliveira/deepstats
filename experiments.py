from uncertainty import anomaly
import pandas as pd

try:
    df = pd.read_csv("bayesian_uncertainty_experiments.csv")
except:
    df = pd.DataFrame()

# df = df.append(anomaly("test_mnist_2labels_1_v1", "bayesian", "mnist", [0,1], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_2labels_2_v1", "bayesian", "mnist", [2,7], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_2lavels_3_v1", "bayesian", "mnist", [8,9], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_mnist_5labels_1_v1", "bayesian", "mnist", [0,2,4,6,8], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_5labels_2_v1", "bayesian", "mnist", [1,3,5,7,9], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_5labels_3_v1", "bayesian", "mnist", [0,1,2,3,4], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_mnist_8labels_1_v1", "bayesian", "mnist", [2,3,4,5,6,7,8,9], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_8labels_2_v1", "bayesian", "mnist", [0,1,3,4,5,6,8,9], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_8labels_3_v1", "bayesian", "mnist", [0,1,2,3,4,5,6,7], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_mnist_2labels_1_v2", "bayesian", "mnist", [0,1], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_2labels_2_v2", "bayesian", "mnist", [2,7], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_2lavels_3_v2", "bayesian", "mnist", [8,9], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_mnist_5labels_1_v2", "bayesian", "mnist", [0,2,4,6,8], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_5labels_2_v2", "bayesian", "mnist", [1,3,5,7,9], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_5labels_3_v2", "bayesian", "mnist", [0,1,2,3,4], 50, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_mnist_8labels_1_v2", "bayesian", "mnist", [2,3,4,5,6,7,8,9], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_8labels_2_v2", "bayesian", "mnist", [0,1,3,4,5,6,8,9], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_mnist_8labels_3_v2", "bayesian", "mnist", [0,1,2,3,4,5,6,7], 75, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv")



# df = df.append(anomaly("test_cifar_2labels_1_v1", "bayesian", "cifar10", [0,1], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_2labels_2_v1", "bayesian", "cifar10", [2,7], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_2lavels_3_v1", "bayesian", "cifar10", [8,9], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_cifar_5labels_1_v1", "bayesian", "cifar10", [0,2,4,6,8], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_5labels_2_v1", "bayesian", "cifar10", [1,3,5,7,9], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_5labels_3_v1", "bayesian", "cifar10", [0,1,2,3,4], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_cifar_8labels_1_v1", "bayesian", "cifar10", [2,3,4,5,6,7,8,9], 75, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_8labels_2_v1", "bayesian", "cifar10", [0,1,3,4,5,6,8,9], 75, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_8labels_3_v1", "bayesian", "cifar10", [0,1,2,3,4,5,6,7], 75, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_cifar_2labels_1_v2", "bayesian", "cifar10", [0,1], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_2labels_2_v2", "bayesian", "cifar10", [2,7], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_2lavels_3_v2", "bayesian", "cifar10", [8,9], 25, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_cifar_5labels_1_v2", "bayesian", "cifar10", [0,2,4,6,8], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_5labels_2_v2", "bayesian", "cifar10", [1,3,5,7,9], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_5labels_3_v2", "bayesian", "cifar10", [0,1,2,3,4], 50, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")


# df = df.append(anomaly("test_cifar_8labels_1_v2", "bayesian", "cifar10", [2,3,4,5,6,7,8,9], 75, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

# df = df.append(anomaly("test_cifar_8labels_2_v2", "bayesian", "cifar10", [0,1,3,4,5,6,8,9], 75, 60))
# df.to_csv("bayesian_uncertainty_experiments.csv")

df = df.append(anomaly("test_cifar_8labels_3_v2", "bayesian", "cifar10", [0,1,2,3,4,5,6,7], 75, 60))
df.to_csv("bayesian_uncertainty_experiments.csv")

