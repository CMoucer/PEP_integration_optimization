###############################################################################
# METHODS AS A FUNCTION OF THE STEP SIZE
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

## EXAMPLES
# oder-2 two stage explicit Runge-Kutta method
alpha = np.zeros(3)
alpha[0], alpha[1], alpha[2] = 5, 4, 1
beta = np.zeros(3)
beta[0], beta[1], beta[2]= 2, 4, 0

## STEP SIZES
gammas = np.linspace(0, 1, 10)

plot_LMM_step_size(L, mu, gammas, alpha, beta, epsilon)
