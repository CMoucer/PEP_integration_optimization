###############################################################################
# GRID FOR CONSISTENT LMM
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

## Import usefull functions
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LMM_G_stability import LMM_contraction, LMM_contraction_simplified, LMM_convergence
from Plot_LMM import plot_LMM_step_size, plot_LMM_condition_number

## Class of functions
L = 1. # smoothness
mu = 0.1 # strong-convexity

## Parameter for binary-search
epsilon = 10**(-3)

## GRID
# number of points :
n = 15
# step size :
gamma = 1/L

x = np.linspace(0., 1, n)
y = np.linspace(-1., 1, n)
X, Y = np.meshgrid(x, y)

Z1 = np.zeros((n,n))
Z2 = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        alpha = [x[i], -(1 + x[i]), 1]
        beta = [y[j], (1 - y[j] - x[i]), 0]
        Z1[i][j] = LMM_contraction(L, mu, gamma, alpha, beta, epsilon) 
        Z2[i][j] = LMM_convergence(L, mu, gamma, alpha, beta, epsilon) 

plt.pcolor(X, Y, np.log(Z1).T)
plt.colorbar()
plt.xlabel('y')
plt.ylabel('x')
plt.title('Contraction')
plt.show()

plt.pcolor(X, Y, np.log(Z2).T)
plt.colorbar()
plt.xlabel('y')
plt.ylabel('x')
plt.title('Convergence')
plt.show()
