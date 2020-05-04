# Advanced Course In Machine Learning
# Exercise 2
# Matrix Factorization Model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import seaborn as sns

#################################################
#                Exercise 2.ab                  #
#################################################

# Function iteration for each iteration with u and v in the main algorithm
# Input : solved vectors, fixed vectors, data, weight and K
# Output : the solved vectors
def iteration(solved, fixed, data, weight, K):
    A = np.dot(fixed, fixed.T) + np.eye(K) * weight
    B = np.dot(data, fixed.T)
    A_inv = np.linalg.inv(A)
    solved = B.dot(A_inv).T
    return solved

# Function prediction
# Input : the columns of u and v
# Output : the prediction of u and v
def prediction(u, v):
    predict = np.dot(u.T, v)
    return predict

# Function MSE
# Input : real_value and pred_value
# Output : the mean squared error of the inputs
def MSE(real_value, pred_value):
    mask = np.nonzero(real_value)
    error = mean_squared_error(real_value[mask], pred_value[mask])
    return error

# Function matrix_factorization
# Input : train set, test set, lambdas, k and number of iterations
# Output : the vectors u, v and the errors
def matrix_factorization(train, test, u_lambda, v_lambda, K, iterations):
    u = np.random.rand(K, 375)
    v = np.random.rand(K, 500)
    test_errors  = []
    train_errors = []

    for x in range(iterations):

        u = iteration(u, v, train, u_lambda, K)
        v = iteration(v, u, train.T, v_lambda, K)

        predictions = prediction(u, v)
        test_error = MSE(test, predictions)
        train_error = MSE(train, predictions)
        test_errors.append(test_error)
        train_errors.append(train_error)

    return u, v, test_errors, train_errors, test_error, train_error


# Function traintest
# Input : data and p
# Output : the train and test
def train_test(data, p):
    train = np.random.rand(375, 500)
    test = data.copy()
    for d in range(test.shape[0]):
        for n in range(test.shape[1]):
            if train[d,n] <= p:
                train[d,n] = test[d,n]
                test[d,n] = 0
            else:
                train[d,n] = 0

    assert np.all(train*test == 0)
    return train, test


#################################################
#                Exercise 2.c                   #
#################################################

# Now we apply the algorithm in the given data.csv
data = pd.read_csv("problem2data.csv", header=None)

# Set the parameters
u_lambda = 0.01
v_lambda = 0.01
K = 8
p_range = [0.1, 0.3, 0.5, 0.7]
test_error_p = []
train_error_p = []

for p in p_range:
    train, test = train_test(data.values, p)
    u, v, test_errors, train_errors, test_error, train_error = matrix_factorization(train, test, u_lambda, v_lambda, K, 100)
    test_error_p.append(test_error)
    train_error_p.append(train_error)
    plt.plot(test_errors, label = 'Test Data')
    plt.plot(train_errors, label = 'Train Data')
    plt.xlabel('Iterations')
    plt.ylabel('MEAN SQUARED ERROR')
    plt.title('Test-Train Mean Squared Error (p = {} and K = 8)'.format(p))
    plt.show()


#################################################
#                Exercise 2.d                   #
#################################################

# Plot 2D for the values of P
# I would use the higher values of p since they make the error smaller
plt.plot(p_range, test_error_p, label = 'Test', linestyle='--', marker='o')
plt.plot(p_range, train_error_p, label = 'Train', linestyle='--', marker='o')
plt.xlabel('p')
plt.ylabel('MSE')
plt.title('Test-Train Mean Squared Error wrt p')
plt.show()


#################################################
#                Exercise 2.e                   #
#################################################

# Now we fix p = 0.1 and experiment with different values of K
p = 0.1
train, test = train_test(data.values, p)
K_range = [2, 8, 20, 80]
test_error_K = []
train_error_K = []

for K in K_range:
    u, v, test_errors, train_errors, test_error, train_error = matrix_factorization(train, test, u_lambda, v_lambda, K, 100)
    test_error_K.append(test_error)
    train_error_K.append(train_error)
    plt.plot(test_errors, label = 'Test')
    plt.plot(train_errors, label = 'Train')
    plt.xlabel('#Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Test-Train Mean Squared Error with p = 0.1 and K = {}'.format(K))
    plt.show()

# Plot 2D of the values of K
# I would use the smallest value of K since the error is smaller
plt.plot(K_range, test_error_K, label = 'Test', linestyle='--', marker='o')
plt.plot(K_range, train_error_K, label = 'Train', linestyle='--', marker='o')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.title('Test-Train Mean Squared Error wrt K')
plt.show()
