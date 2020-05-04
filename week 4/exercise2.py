# Advanced Course in Machine Learning
# Exercise 2
# adaboost

from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Definition of the adaboost algorithm
# Parameters : X vector input
#              y vector output
#              M loop range
# Return : errors and loss function
def adaboost(X, Y, M):
    N = len(Y)

    # List for the weights
    weight_vector = np.ones(N)/N
    weight_vector_list = []
    weight_vector_list.append(weight_vector)

    epsilon_m = []  # estimation of errors
    beta_m = []  # estimation of weights
    f_m = [] # estimation for prediction

    ensemble_error_list = []
    exponential_loss_list = []
    ensemble_pred = 0

    for m in range(M):
        estimator = DecisionTreeClassifier(max_depth = 1, max_leaf_nodes=2)
        estimator.fit(X, Y, sample_weight=weight_vector)
        y_pred = estimator.predict(X)
        f_m.append(y_pred)

        miss = (y_pred != Y)

        estimator_error = np.mean(np.average(miss, weights=weight_vector, axis=0))
        epsilon_m.append(estimator_error.copy())

        estimator_weight = 1/2 * np.log((1. - estimator_error)/estimator_error)
        beta_m.append(estimator_weight)

        ensemble_pred += estimator_weight*y_pred
        miss_ensemble = (np.sign(ensemble_pred) != Y)
        ensemble_error = np.mean(np.average(miss_ensemble, weights=weight_vector, axis=0))
        ensemble_error_list.append(ensemble_error.copy())

        exponential_loss = np.sum(np.exp(-1.*Y*ensemble_pred))
        exponential_loss_list.append(exponential_loss.copy())

        weight_vector *= np.exp(-1.*estimator_weight*y_pred*Y)
        weight_vector_list.append(weight_vector)

    f_m = np.asarray(f_m)
    epsilon_m = np.asarray(epsilon_m)
    beta_m = np.asarray(beta_m)
    ensemble_error_list = np.asarray(ensemble_error_list)
    exponential_loss_list = np.asarray(exponential_loss_list)
    weight_vector_list = np.asarray(weight_vector_list)

    return epsilon_m, ensemble_error_list, exponential_loss_list, ensemble_error


# Now we read the provided file in Moodle
data = pd.read_csv("problem2.csv", header=None)
X = data.drop(data.columns[2], axis=1)
Y = data[2]

# We apply Adaboost algorithm to the data
epsilon_m, ensemble_error_list, exponential_loss_list, emsemble_error = adaboost(X, Y, 1000)

# Plot of the error of individual weak learner
plt.plot(range(len(epsilon_m)), epsilon_m)
plt.xlabel('M')
plt.ylabel('epsilon_m')
plt.title('Error of individual')
plt.show()

# Plot of the error of the ensemble
plt.plot(range(len(ensemble_error_list)), ensemble_error_list)
plt.xlabel('M')
plt.ylabel('epsilon_m')
plt.title('Error of the ensemble')
plt.show()

# Plot of the error of the exponential loss
plt.plot(range(len(exponential_loss_list)), exponential_loss_list)
plt.xlabel('M')
plt.ylabel('epsilon_m')
plt.title('Exponential loss')
plt.show()
