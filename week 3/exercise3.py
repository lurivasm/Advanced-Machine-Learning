# Advanced Course In Machine Learning
# Exercise 3
# Spectral Clustering

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import scipy as sp

# Read the data
data = pd.read_csv("problem3data.csv", sep=",", header=None)

# Compute the distance matrix
Dmatrix_values = sp.spatial.distance.cdist(data, data, metric='euclidean')
Dmatrix = pd.DataFrame(Dmatrix_values)
print('\tThe distance matrix of the original data:')
print(Dmatrix)

# K means algorithm with K = 2
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Plot wrt k means
sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data for Exercise 3 with k means (k = 2)')
plt.show()

#################################################
#                Exercise 3.a                   #
#################################################

# Adjacency matrices : W1 and W2
e = 0.5

# W1(i,j) = 1 if d(i,j) ≤ e
W1 = Dmatrix.apply(lambda x : x <= e)
print('\tFirst adjacent matrix, e < {}:'.format(e))
print(W1)

# 8 closest neighbors
A = 8
W2 = pd.DataFrame(np.zeros((120,120)), dtype=bool)
neighbors = pd.DataFrame()
for i in range(120):
    neighbors.insert(i, i, Dmatrix.nsmallest(A+1, i).iloc[:,i].drop(Dmatrix.index[i]).index)
    for j in range(120):
        if (j in neighbors[i].values):
            W2.at[i,j] = 1
            W2.at[j,i] = 1
print('\tSecond adjacent matrix, A = {} closest neighbors:'.format(A))
print(W2)


#################################################
#                Exercise 3.b                   #
#################################################

# Diagonal matrices for W1 and W2
D1 = pd.DataFrame(np.zeros((120,120)))
D2 = pd.DataFrame(np.zeros((120,120)))
for i in range(120):
    D1.at[i,i] = W1.iloc[i,:].sum()
    D2.at[i,i] = W2.iloc[i,:].sum()

# Laplacian for W1 and W2
L1 = D1 - W1
L2 = D2 - W2

# Eigenvalues and eigenvector for W1
eigval_1, eigvect_1 = np.linalg.eig(L1)
eigenvalue1 = pd.DataFrame(eigval_1, columns=['eigval']).sort_values(by=['eigval'], ascending=True)
eigenvector1 = pd.DataFrame(eigvect_1).transpose().reindex(eigenvalue1.index)
eigenvector1 = eigenvector1.reset_index(drop=True)
print('\tFour smallest eigenvectors, e < {}:'.format(e))
print(eigenvector1)

# Eigenvalues and eigenvector for W2
eigval_2, eigvect_2 = np.linalg.eig(L2)
eigenval_2 = pd.DataFrame(eigval_2, columns=['eigval']).sort_values(by=['eigval'], ascending=True)
eigenvector2 = pd.DataFrame(eigvect_2).transpose().reindex(eigenval_2.index)
eigenvector2 = eigenvector2.reset_index(drop=True)
print('\tFour smallest eigenvectors, A = {} closest neighbors:'.format(A))
print(eigenvector2)

# There is a big change after the 60th that can be the division
# of the n_clusters
# It is easier to see in eigenvectors 2 and 3

# Plot of eigenvectors of W1
plt.plot(range(120), eigenvector1.iloc[0,:])
plt.plot(range(120), eigenvector1.iloc[1,:])
plt.plot(range(120), eigenvector1.iloc[2,:])
plt.plot(range(120), eigenvector1.iloc[3,:])
plt.legend(['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4'])
plt.xlabel('x')
plt.ylabel('Eigenvectors')
plt.title('First smallest eigenvetors, e < {}'.format(e))
plt.show()

# Plot of eigenvectors of W2
plt.plot(range(120), eigenvector2.iloc[0,:])
plt.plot(range(120), eigenvector2.iloc[1,:])
plt.plot(range(120), eigenvector2.iloc[2,:])
plt.plot(range(120), eigenvector2.iloc[3,:])
plt.legend(['Eigenvector 1', 'Eigenvector 2', 'Eigenvector 3', 'Eigenvector 4'])
plt.xlabel('x')
plt.ylabel('Eigenvectors')
plt.title('First smallest eigenvetors, A = {} closest neighbors'.format(A))
plt.show()


#################################################
#                Exercise 3.c                   #
#################################################

# Create a new representation Y 120x4
M = 4
Y1 = pd.DataFrame(eigenvector1.iloc[0:M,:].transpose())
Y2 = pd.DataFrame(eigenvector2.iloc[0:M,:].transpose())
print('\tOriginal data:')
print(data)
print('\tNew data for Y1, e < {}:'.format(e))
print(Y1)
print('\tNew data for Y2, A = {} closest neighbors:'.format(A))
print(Y2)

#################################################
#                Exercise 3.d                   #
#################################################

# We apply PCA to Y1 and Y2 to reduce to 2dim
PCA1 = PCA(n_components=2)
principal_1 = pd.DataFrame(data=PCA1.fit_transform(Y1.values))
sns.scatterplot(principal_1.iloc[:,0], principal_1.iloc[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of PCA Y1, e < {}'.format(e))
plt.show()

PCA2 = PCA(n_components=2)
principal_2 = pd.DataFrame(data=PCA2.fit_transform(Y2.values))
sns.scatterplot(principal_2.iloc[:,0], principal_2.iloc[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of PCA Y2, A = {} closest neighbors'.format(A))
plt.show()

#################################################
#                Exercise 3.e                   #
#################################################

# Now we apply kmeans to the new data Y1, Y2

kmeans1 = KMeans(n_clusters=2, random_state=0).fit(principal_1.values)

sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans1.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of PCA Kmeans1, e < {}'.format(e))
plt.show()

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(principal_2.values)

sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans2.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of PCA Kmeans2, A = {} closest neighbors'.format(A))
plt.show()
