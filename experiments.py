from uncertainty import anomaly
import pandas as pd

try:
    df = pd.read_csv("bayesian_uncertainty_experiments_mnist.csv")
except:
    df = pd.DataFrame()


df = df.append(anomaly("test_mnist_4labels_2unknown_1.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [0, 1, 4, 8], unknown_labels = [7, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_1.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [0, 1, 4, 8], unknown_labels = [7, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_1.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [0, 1, 4, 8], unknown_labels = [7, 9]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_2.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 2, 8, 7], unknown_labels = [3, 6]))
df = df.append(anomaly("test_mnist_4labels_2unknown_2.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 2, 8, 7], unknown_labels = [3, 6]))
df = df.append(anomaly("test_mnist_4labels_2unknown_2.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 2, 8, 7], unknown_labels = [3, 6]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_3.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 8, 3, 2], unknown_labels = [5, 0]))
df = df.append(anomaly("test_mnist_4labels_2unknown_3.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 8, 3, 2], unknown_labels = [5, 0]))
df = df.append(anomaly("test_mnist_4labels_2unknown_3.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 8, 3, 2], unknown_labels = [5, 0]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_4.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [5, 3, 7, 8], unknown_labels = [2, 4]))
df = df.append(anomaly("test_mnist_4labels_2unknown_4.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [5, 3, 7, 8], unknown_labels = [2, 4]))
df = df.append(anomaly("test_mnist_4labels_2unknown_4.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [5, 3, 7, 8], unknown_labels = [2, 4]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_5.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 5, 0, 6], unknown_labels = [1, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_5.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 5, 0, 6], unknown_labels = [1, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_5.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 5, 0, 6], unknown_labels = [1, 3]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_6.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [3, 0, 6, 9], unknown_labels = [1, 5]))
df = df.append(anomaly("test_mnist_4labels_2unknown_6.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [3, 0, 6, 9], unknown_labels = [1, 5]))
df = df.append(anomaly("test_mnist_4labels_2unknown_6.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [3, 0, 6, 9], unknown_labels = [1, 5]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_7.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 6, 1, 8], unknown_labels = [4, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_7.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 6, 1, 8], unknown_labels = [4, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_7.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 6, 1, 8], unknown_labels = [4, 3]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_8.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 0, 2], unknown_labels = [5, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_8.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 0, 2], unknown_labels = [5, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_8.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 0, 2], unknown_labels = [5, 9]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_9.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [2, 5, 7, 9], unknown_labels = [8, 0]))
df = df.append(anomaly("test_mnist_4labels_2unknown_9.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [2, 5, 7, 9], unknown_labels = [8, 0]))
df = df.append(anomaly("test_mnist_4labels_2unknown_9.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [2, 5, 7, 9], unknown_labels = [8, 0]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_10.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [1, 8, 6, 2], unknown_labels = [4, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_10.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [1, 8, 6, 2], unknown_labels = [4, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_10.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [1, 8, 6, 2], unknown_labels = [4, 9]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_11.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 1, 0, 7], unknown_labels = [9, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_11.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 1, 0, 7], unknown_labels = [9, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_11.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 1, 0, 7], unknown_labels = [9, 3]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_12.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 6, 2, 8], unknown_labels = [5, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_12.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 6, 2, 8], unknown_labels = [5, 3]))
df = df.append(anomaly("test_mnist_4labels_2unknown_12.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 6, 2, 8], unknown_labels = [5, 3]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_13.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 5, 7, 1], unknown_labels = [8, 4]))
df = df.append(anomaly("test_mnist_4labels_2unknown_13.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 5, 7, 1], unknown_labels = [8, 4]))
df = df.append(anomaly("test_mnist_4labels_2unknown_13.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 5, 7, 1], unknown_labels = [8, 4]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_14.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 0, 5, 9], unknown_labels = [3, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_14.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 0, 5, 9], unknown_labels = [3, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_14.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 0, 5, 9], unknown_labels = [3, 2]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_15.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 3, 5, 1], unknown_labels = [8, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_15.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 3, 5, 1], unknown_labels = [8, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_15.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [7, 3, 5, 1], unknown_labels = [8, 2]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_16.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 7, 3, 0], unknown_labels = [5, 6]))
df = df.append(anomaly("test_mnist_4labels_2unknown_16.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 7, 3, 0], unknown_labels = [5, 6]))
df = df.append(anomaly("test_mnist_4labels_2unknown_16.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 7, 3, 0], unknown_labels = [5, 6]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_17.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 3, 8, 4], unknown_labels = [0, 7]))
df = df.append(anomaly("test_mnist_4labels_2unknown_17.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 3, 8, 4], unknown_labels = [0, 7]))
df = df.append(anomaly("test_mnist_4labels_2unknown_17.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [9, 3, 8, 4], unknown_labels = [0, 7]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_18.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 9, 8], unknown_labels = [1, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_18.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 9, 8], unknown_labels = [1, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_18.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [6, 4, 9, 8], unknown_labels = [1, 2]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_19.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 2, 3, 7], unknown_labels = [0, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_19.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 2, 3, 7], unknown_labels = [0, 9]))
df = df.append(anomaly("test_mnist_4labels_2unknown_19.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [8, 2, 3, 7], unknown_labels = [0, 9]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)

df = df.append(anomaly("test_mnist_4labels_2unknown_20.1", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 8, 7, 3], unknown_labels = [9, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_20.2", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 8, 7, 3], unknown_labels = [9, 2]))
df = df.append(anomaly("test_mnist_4labels_2unknown_20.3", network='bayesian', dataset="mnist", nb_epochs = 100, acc_threshold = 0.98, inside_labels = [4, 8, 7, 3], unknown_labels = [9, 2]))
df.to_csv("bayesian_uncertainty_experiments_mnist.csv", index=False)


# df = df.append(anomaly("test_mnist_2labels_1_v1", "bayesian", "mnist", [0,1], 25, 100))
# df.to_csv("bayesian_uncertainty_experiments.csv", index=False)

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

#df = df.append(anomaly("test_cifar_8labels_3_v2", "bayesian", "cifar10", [0,1,2,3,4,5,6,7], 75, 60))
#df.to_csv("bayesian_uncertainty_experiments.csv")

