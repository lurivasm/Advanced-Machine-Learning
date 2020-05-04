#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

###################
## Exercise 3.a) ##
###################

# First we read the data
dataFile = "ex_1_data.csv"
data = pd.read_csv(dataFile, header=None)
original_data = data.copy()

# We can see that N = 200 (rows) and D = 5 (columns)

# Standardize the data
data = pd.DataFrame(StandardScaler().fit_transform(data))
dataCopy = data.copy() # Do a comparison of the phases with the SKlearn PCA to double check


###################
## Exercise 3.b) ##
###################

# Covariance Matrix
cov_matrix = data.cov()
print('\nThe covariance matrix is: \n')
print(cov_matrix)

# Eigenvalues and eigenvectorors
eigenvalue, eigenvector = np.linalg.eig(cov_matrix)
eigenvalues_ = pd.DataFrame(eigenvalue, columns=['eigenvals'])
eigenvectors_ = pd.DataFrame(eigenvector)

# Print eigenvalues in descendent order
eigenvalues_ = eigenvalues_.sort_values('eigenvals', ascending=False)
print('\nEigenvalues\n')
print(eigenvalues_)

# Print matrix of eigenvectors (= columns)
print('\nEigenvectors\n')
print(eigenvectors_)


###################
## Exercise 3.c) ##
###################

# Order vectors wrt values and transpose (=rows)
eigenvectors_ = eigenvectors_.transpose()
eigenvectors_ = eigenvectors_.reindex(eigenvalues_.index)
eigenvectors_ = eigenvectors_.reset_index(drop=True)

# Project the two max eigenvals eigenvectors
projections = list()
for i in range(len(eigenvectors_.iloc[0])):
 projections.append(pd.DataFrame(np.dot(eigenvectors_.iloc[0:(i+1),:],data.transpose())))

for i in range(len(projections)):
  projections[i] = projections[i].transpose()

# Plot two max projections
sns.scatterplot(projections[4].iloc[:,0], projections[4].iloc[:,1]).plot()
plt.title('Eigenvector projection of csv file')
plt.xlabel('First eigenvector projection')
plt.ylabel('Second eigenvector projection')
plt.show()


###################
## Exercise 3.d) ##
###################

for i in range(len(projections)):
 projections[i] = pd.DataFrame(projections[i], columns=None)

# Reconstructing error from L=1 till L=D (=5 in this case)
reconstructing_error = list()
for i in range(len(eigenvector)):
 reconstructing_error.append(np.dot(eigenvectors_.iloc[0:(i+1),:].transpose(), projections[i].transpose()).transpose())

# Errors in matrix
losses = list()
for rec in reconstructing_error:
 lossMatrix = original_data.sub(rec)
 lossMatrix = lossMatrix**2
 losses.append(lossMatrix.values.sum())

# Plot of squared error
sns.scatterplot(range(1,6), losses).plot()
plt.title('Reconstructing the squared error')
plt.xlabel('Number of components')
plt.ylabel('Error loss')
plt.show()
