# Advanced Course In Machine Learning
# Exercise 1
# Empirical Risk

import numpy as np
import matplotlib.pyplot as plt


#################################################
#                Exercise 1.b                   #
#################################################

# Function MonteCarloRisk
# Params : alfa is the parameter for the model function alfa*x
#          M is the size of the vectors x and y
# Return : the risk calculated with Monte Carlo integration
def MonteCarloRisk(alfa, M):
    x = np.random.uniform(-3 ,3 , M)
    ini = np.random.uniform(-0.5, 0.5, M)
    y = 2*x + ini

    loss = (y - alfa*x)**2
    risk = sum(loss)/M
    return risk

#################################################
#                Exercise 1.c                   #
#################################################

# We set the alfa and put the risk in function of M
# The number of perfect samples would be between 10000 and 100000
# by checking the plots in the folder
alfa = 1.5
risks = list()
M_plot = range(1, 10001)

for M in M_plot:
    risks.append(MonteCarloRisk(alfa, M))

plt.plot(M_plot, risks)
plt.xlabel('Risk with alfa 1.5')
plt.ylabel('M')
plt.title('Risk with ' + str(M_plot[-1]) + ' number of samples')
plt.show()
