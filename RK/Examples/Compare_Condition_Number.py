###############################################################################
# COMPARE SCHEMES AS A FUNCTION OF THE CONDITION NUMBER
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality
from RK_G_stability import G_stability_RK, G_stability_RK_optimality

###############################################################################

## CLASS OF FUNCTIONS
L = 1. # smoothness
# condition numbers :
kc = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
mus = L/kc

## SCHEMES TO STUDY
#schemes = ['Euler explicit', 'Two stage 1', 'Euler implicit', 'Two stage explicit', 'Trapezoidal']
schemes = ['Euler explicit', 'Two stage 1']
# Step size
gamma = 1/L

# Compute the contraction rate for different values of the condition number
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
    ratesB = np.zeros(len(mus))
    ratesBopt = np.zeros(len(mus))
    for i in range(len(mus)):
        ratesB[i] = B_stability(L, mus[i], gamma, alpha, beta)
        ratesBopt[i] = B_stability_optimality(L, mus[i], gamma, alpha, beta)
    plt.plot(L/mus, ratesB, '-x', label=scheme + ' contraction')
    plt.plot(L/mus, ratesBopt, '-x', label=scheme + ' convergence')
    
# Plot the function
plt.plot(L/mus, np.ones(len(mus)), '--')
plt.xlabel('Condition number')
plt.ylabel('Rate')
plt.legend()
plt.semilogy()
plt.semilogx()
plt.show()