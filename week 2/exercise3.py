# Advanced Course In Machine Learning
# Exercise 3
# Lagrange Multipliers

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Iterative and recursive method for Lagrange Multipliers
# Lagragian := L(x, y, lambda) = x^2 + y^2 + lambda*(3x - y + 2)
# Gradient of Lagragian := (dL/dx, dL/dy) = (2x + 3lambda, 2y - lambda)

Lambda = 0
x, y = sym.symbols('x, y')

for j in range(1, 100):
    solution = sym.solve([2*x + 3*Lambda, 2*y - Lambda])
    condition = 3*solution[x] - solution[y] + 2

    if(condition <= 0):
        Lambda = Lambda - (1/2**j)
    else:
        Lambda = Lambda + (1/2**j)

print('The solution is:\n(x, y, lambda) = (' + str(solution[x]) + ', ' + str(solution[y]) + ', ' + str(Lambda) + ')')

# It is the same result as in 3.a)
