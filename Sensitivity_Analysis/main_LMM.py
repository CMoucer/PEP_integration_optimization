###############################################################################
# SENSITIVITY ANALYSIS OF LMM METHODS
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from sensitivity_analysis_LMM import LMM_contraction_N, LMM_contraction_restarted

## Class of functions
L = 1. # smoothness

## Parameter for binary-search
epsilon = 10**(-3)
mu = 0.1
kc = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
mus = L/kc

## SCHEME CHOICE : 
scheme = 'Nesterov'

###############################################################################
## EXAMPLES : COMPARE CONTRACTION RATE TO ITS RESTARTED VERSION 
print('Comparison between contraction rate and restarted version for {}'.format(scheme))

rate10 = np.zeros(len(mus))
rate11 = np.zeros(len(mus))
rate20 = np.zeros(len(mus))

for i in range(len(mus)):
    if scheme == 'Nesterov':
        kappa = (np.sqrt(L / mus[i]) - 1) / (np.sqrt(L / mus[i]) + 1)
        alpha = [kappa, -(1 + kappa), 1]
        beta = [- kappa, (1 + kappa), 0]
        gamma = 1 / L
    if scheme == 'TMM':
        rho = 1 - np.sqrt(mus[i] / L)
        b = rho ** 2 / (2 - rho)
        a = rho ** 2 / ((1 + rho) * (2 - rho))
        alpha = [b, -(1 + b), 1]
        beta = [-a, (1 + a), 0]
        gamma = (1 + rho) / L
    if scheme == 'optimized':
        kappa = 1-np.sqrt(mus[i]/L)
        alpha = [kappa, -(1 + kappa), 1]
        beta = [- (1 + kappa)/2, (3 - kappa)/2, 0]
        gamma = 3/(2*L)
    rate10[i], N = LMM_contraction_N(L, mus[i], gamma, alpha, beta, epsilon,iteration = 3)
    rate11[i] = LMM_contraction_restarted(L, mus[i], gamma, alpha, beta, N, rate10[i], iteration = 4 , tolerance = 0) 
    rate20[i], _ = LMM_contraction_N(L, mus[i], gamma, alpha, beta, epsilon, iteration = 4)

## PLOT THE COMPARISON
plt.plot(L/mus, rate10, '-o',label = 'iteration 1')
plt.plot(L/mus, rate11, '-x',label = 'restarted iteration 2')
plt.plot(L/mus, rate20, '-x',label = 'iteration 2')
plt.plot(L/mus, np.ones(len(mus)), '--') 

plt.xlabel('condition number')
plt.ylabel('contraction rate')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.show()


###############################################################################
### COMPARE AT SEVERAL ITERATES
print('Compare contraction rate for {} at different iterations'.format(scheme))

# iterations to observe
iterations = np.array([3, 5, 8, 10, 12])

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
if scheme == 'optimized':
    kappa = 1-np.sqrt(mus/L)
    alpha = [kappa, -(1 + kappa), 1]
    beta = [- (1 + kappa)/2, (3 - kappa)/2, 0]
    gamma = 3/(2*L)

rate = np.zeros(len(iterations))
for i in range(len(iterations)):
    rate[i], _ = LMM_contraction_N(L, mu, gamma, alpha, beta, epsilon, iteration = iterations[i])

## PLOT
plt.plot(iterations, rate, '-x', label=scheme)
plt.plot(iterations, np.ones(len(iterations))*(1-mu/L)**2, '-x', label='Euler')
plt.plot(iterations, np.ones(len(iterations)), '--')

plt.xlabel('iterations')
plt.ylabel('Contraction rate')
plt.legend()
plt.show()
