###############################################################################
# METHODS AS A FUNCTION OF THEIR CONDITION NUMBER
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

## Import usefull functions
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LMM_G_stability import LMM_contraction, LMM_contraction_simplified, LMM_convergence

## Class of functions
L = 1. # smoothness
kc = np.array([10,20,50,100,200, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
mus = L/kc # strong-convexity

## Parameter for binary-search
epsilon = 10**(-3)

## EXAMPLES
scheme = 'Nesterov'

rate_conv = np.zeros(len(mus))
rate_cont = np.zeros(len(mus))
rate_cont_spl = np.zeros(len(mus))
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
    if scheme == 'Designed 1' :
        kappa = 1-np.sqrt(mus[i]/L)
        alpha = [kappa, -(1 + kappa), 1]
        beta = [- (1 + kappa)/2, (3 - kappa)/2, 0]
        gamma = 2/(L+mus[i])*(kappa+1)/(3-kappa)
    if scheme == 'Designed 2':
        nu = (1 - np.sqrt(mus[i]/L))
        gamma = 2*(1-np.sqrt(mus[i]/L))/(L+mus[i])
        kappa = (3*nu - 1)/(nu + 1)
        alpha = [kappa, -(1 + kappa), 1]
        beta = [- (1 + kappa)/2, (3 - kappa)/2, 0]
    rate_conv[i] = LMM_convergence(L, mus[i], gamma, alpha, beta, epsilon)
    rate_cont[i] = LMM_contraction(L, mus[i], gamma, alpha, beta, epsilon)
    rate_cont_spl[i] = LMM_contraction_simplified(L, mus[i], gamma, alpha, beta, epsilon)

plt.plot(L/mus, rate_conv, '-x', label='Convergence rate')
plt.plot(L/mus, rate_cont, '-x', label='Contraction rate')
plt.plot(L/mus, rate_cont_spl, '-x', label='Contraction rate (simplified)')

plt.legend()
plt.plot(L/mus, np.ones(len(mus)), '--')
plt.semilogy()
plt.semilogx()
plt.xlabel('condition number')
plt.ylabel('rate')
plt.show()
    
    
    
