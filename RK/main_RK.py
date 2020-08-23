###############################################################################
# COMPUTE CONTRACTION AND CONVERGENCE RATES FOR A RK SCHEME
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality
from RK_G_stability import G_stability_RK, G_stability_RK_optimality

## Class of functions
L = 1. # smoothness
mu = 0.1 # strong-convexity

## Parameter for binary-search
epsilon = 10**(-3)

## EXAMPLES
# name of the scheme
scheme = 'Euler explicit'
# step size
gamma = 1/L

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
    
## COMPUTE THE RATES
rho_B = B_stability(L, mu, gamma, alpha, beta)
print('Contraction rate for B-stability of {} is equal to : '.format(scheme), rho_B)

nu_B = B_stability_optimality(L, mu, gamma, alpha, beta)
print('Convergence rate for B-stability to optimality of {} is equal to : '.format(scheme), nu_B)

rho_G = G_stability_RK(L, mu, gamma, alpha, beta, epsilon)
print('Contraction rate for G-stability of {} is equal to : '.format(scheme), rho_G)

nu_G = G_stability_RK_optimality(L, mu, gamma, alpha, beta, epsilon)
print('Convergence rate for G-stability to optimality of {} is equal to : '.format(scheme), nu_G)









    
    