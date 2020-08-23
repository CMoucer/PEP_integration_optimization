###############################################################################
# COMPUTE CONTRACTION AND CONVERGENCE RATES FOR A LMM SCHEME
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

## Import usefull functions
from LMM_G_stability import LMM_contraction, LMM_contraction_simplified, LMM_convergence

## Class of functions
L = 1. # smoothness
mu = 0.1 # strong-convexity

## Parameter for binary-search
epsilon = 10**(-3)

## EXAMPLES
scheme = 'Nesterov'

if scheme == 'Nesterov':
    kappa = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
    alpha = [kappa, -(1 + kappa), 1]
    beta = [- kappa, (1 + kappa), 0]
    gamma = 1 / L
if scheme == 'TMM':
    rho = 1 - np.sqrt(mu / L)
    b = rho ** 2 / (2 - rho)
    a = rho ** 2 / ((1 + rho) * (2 - rho))
    alpha = [b, -(1 + b), 1]
    beta = [-a, (1 + a), 0]
    gamma = (1 + rho) / L

## Print contraction and convergence rates
## COMPUTE THE RATES
rho_contraction = LMM_contraction(L, mu, gamma, alpha, beta, epsilon)
print('Contraction rate for a quadratic Lyapunov of {} is equal to : '.format(scheme), rho_contraction)

nu_contraction = LMM_contraction_simplified(L, mu, gamma, alpha, beta, epsilon)
print('Contraction rate for a quadratic simplified Lyapunov of {} is equal to : '.format(scheme), nu_contraction)

rho_convergence = LMM_convergence(L, mu, gamma, alpha, beta, epsilon)
print('Convergence rate for a quadratic Lyapunov of {} is equal to : '.format(scheme), rho_convergence)



