###############################################################################
# PLOT A GRID FOR CONSISTENT TWO STAGE RK METHOD (explicit)
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality

###############################################################################

## CLASS OF FUNCTIONS
L = 1. # smoothness
mu = 0.1 # strong-convexity

## SCHEME
alpha = np.zeros((2,2))
alpha[1][0] = 0.5
beta = 0.5*np.ones(2)
# step size : 
gamma = 1/L

## MESHGRID
# number of points :
n = 15
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

Z1 = np.zeros((n,n))
Z2 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        alpha = np.zeros((2,2))
        alpha[1][0] = y[j]
        beta = np.zeros(2)
        beta[0] = x[i]
        beta[1] = 1 - x[i]
        Z1[i][j] = B_stability(L, mu, gamma, alpha, beta)
        Z2[i][j] = B_stability_optimality(L, mu, gamma, alpha, beta)
        
plt.pcolor(X, Y, np.log(Z1).T)
plt.colorbar()
plt.xlabel('nu')
plt.ylabel('alpha')
plt.title('Contraction')
plt.show()

plt.pcolor(X, Y, np.log(Z2).T)
plt.colorbar()
plt.xlabel('nu')
plt.ylabel('alpha')
plt.title('Convergence')
plt.show()



