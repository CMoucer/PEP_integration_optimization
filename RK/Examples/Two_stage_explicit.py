###############################################################################
# TWO STAGE EXPLICIT STUDY
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality
from RK_G_stability import G_stability_RK, G_stability_RK_optimality
from Plot_RK import plot_RK_step_size, plot_RK_condition_number

###############################################################################

## CLASS OF FUNCTIONS
L = 1. # smoothness
mu = 0.1 # strong-convexity

## SCHEME
alpha = np.zeros((2,2))
alpha[1][0] = 0.5
beta = 0.5*np.ones(2)
# step size : 
gammas = np.linspace(0.5/L, 4./L, 5)

## BINARY SEARCH
epsilon = 10**(-3)

plot_RK_step_size(L, mu, gammas, alpha, beta, epsilon)


