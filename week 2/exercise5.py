# Advanced Course In Machine Learning
# Exercise 5
# Stochastic GRadient Descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

########################################################
#                  SGD Library                         #
########################################################
# Function nextBatch
# It yields a tuple of the next batch of x and y
def nextBatch(x, y, batchSize):
    for i in np.arange(0, x.shape[0], batchSize):
        yield (x[i:i + batchSize], y[i:i + batchSize])

# Function computeError
# It returns the difference between the theorical y and
# the output y of the function f(x,theta)
def computeError (theta, x, y):
    y_theorical = np.asmatrix(y)
    y_function = np.dot(theta, x.transpose())
    error = y_function - y_theorical
    return error

# Function computeLoss
# It returns the squared error (the loss)
def computeLoss (error):
    loss = np.mean(np.square(error))
    return loss

# Function computeGradient
# It returns the formula of the slides
def computeGradient (error, x):
    N = len(x.iloc[:,0])
    sumInter = np.dot(error, x)
    return 2*sumInter/N

#################################################
#                Exercise 5.a                   #
#################################################

# Read the data file
dataFile = "problem5data.csv"
data = pd.read_csv(dataFile, header=None)

# x is a matrix 2x500 (two first columns)
x = data.drop(data.columns[2], axis=1)
# y is a vector (third column)
y = data[2]

# Split into training and test sets for later validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


###################################################
#           Main Loop of the algorithm            #
###################################################

# It stops because it has a fixed number of iterations but we
# could stablish the RM conditions for the convergence

# Initial Values for variables
batchSize = 100 # M loop range
epochs = 100
alfa_zero = 0.5
alfa_t = alfa_zero
beta = 0.0005
t = 0
samples = 0 # Number of samples

# Initial value for theta
theta = np.random.uniform(size=(x_train.shape[1],))
theta = np.asmatrix(theta)

# Arrays for storing the losses
losses = []
loss_validation = []
loss_validation_x = []


# Main loop
for epoch in np.arange(0, epochs):
    # initialize the total loss for the epoch
    epochLoss = []

    for (batchx, batchY) in nextBatch(x, y, batchSize):

        # Compute errors, losses and gradient
        error = computeError(theta, batchx, batchY)
        loss = computeLoss(error)
        epochLoss.append(loss)
        gradient = computeGradient(error, batchx)

        # Update parameters for next step
        theta += -alfa_t * gradient
        t = t + 1
        samples += batchSize

        # Deterministic schedule for the step size
        alfa_t = alfa_zero / (1 + alfa_zero * beta * t)

        # Evaluate test loss every 100 samples
        if (samples % 100 == 0):
            error_test = computeError(theta, x_test, y_test)
            loss_test = computeLoss(error_test)
            loss_validation.append(loss_test)
            loss_validation_x.append(samples)

    losses.append(np.average(epochLoss))

print('The resulting gradient is :')
print(gradient)



##################################################
#               Exercise 5.b and 5.c             #
##################################################

# Plot for the progress
plt.plot(range(len(epochLoss)), epochLoss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Progress of SGD for M = ' + str(batchSize))
plt.show()

# Plot for the validation losses
plt.plot(loss_validation_x, loss_validation)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Validation loss of SGD M = ' + str(batchSize))
plt.show()

# Plot for the training losses
plt.plot(range(len(losses)), losses)
plt.xlabel('Iterations (epochs)')
plt.ylabel('Loss')
plt.title('Training loss of SGD for M = ' + str(batchSize))
plt.show()

# We can see that the results for M=1 and M=100 are almost equal
# but the worst case is always M=10
