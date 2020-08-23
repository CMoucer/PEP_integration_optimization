###############################################################################
# COMPARE SCHEMES AS A FUNCTION OF THE STEP SIZE
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality
from RK_G_stability import G_stability_RK, G_stability_RK_optimality


##############################################################################

## CLASS OF FUNCTIONS
L = 1. # smoothness
mu = 0.1 # strong-convexity

## SCHEMES TO STUDY
#schemes = ['Euler explicit', 'Two stage 1', 'Euler implicit', 'Two stage explicit', 'Trapezoidal']
schemes = ['Euler explicit','Two stage 1','Euler implicit', 'Two stage explicit', 'Trapezoidal']

# Step size
gammas = np.linspace(0,5, 20)

# Compute the contraction rate for different values of the step size
for scheme in schemes:
    print(scheme)
    if scheme == 'Two stage 1':
        alpha = np.zeros((2, 2))
        alpha[1][0] = .5
        beta = np.zeros(2)
        beta[1] = 1.
    if scheme == 'Euler explicit':
        alpha = np.zeros((1, 1))
        beta = np.zeros(1)
        beta[0] = 1.
    if scheme == 'Euler implicit':
        alpha = np.zeros((1, 1))
        alpha[0][0] = 1.
        beta = np.zeros(1)
        beta[0] = 1.
    if scheme == 'Two stage explicit':
        alpha = np.zeros((2, 2))
        alpha[1][0] = .5
        beta = .5 * np.ones(2)
    if scheme == 'Trapezoidal':
        alpha = np.zeros((2, 2))
        alpha[1][0] = 1.
        beta = .5 * np.ones(2)
    rates = np.zeros(len(gammas))
    u=0.
    for i in range(len(gammas)):
        if rates[i-1] >= 5.:
            if u == 0:
                u = gammas[i-1]
            rates[i] = B_stability(L, mu, u, alpha, beta)
        else:
            rates[i] = B_stability(L, mu, gammas[i], alpha, beta)
    plt.plot(gammas, rates, '-x', label=scheme)
    

plt.plot(gammas, np.ones(len(gammas)), '--')
plt.xlabel('Step size')
plt.ylabel('Contraction rate')
plt.legend()
plt.ylim(0.4, 1.5)
plt.show()