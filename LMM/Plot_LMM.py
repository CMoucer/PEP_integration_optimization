###############################################################################
# PLOT LMM RATES AS A FUNCTION OF THE CONDITION NUMBER AND OF THE STEP SIZE
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from LMM_G_stability import LMM_contraction, LMM_contraction_simplified, LMM_convergence


def plot_LMM_step_size(L : float, mu : float, gammas : np.ndarray, alpha, beta, epsilon=10**(-3), simplified = True, opt=True, contraction = True):
    """
    Compute contraction and convergence rates in terms of B and G-stability for a Runge-Kutta scheme
    as a function of the step size gammas
    Parameters
    ----------
    L : float
      smoothness parameter for the class of functions
    
    mu : float
      strong-convexity parameter for the class of functions
    
    gammas : np.ndarray, shape (n,)
      step size of the method
    
    alpha : ndarray, shape (s,)
      internal step size
    
    beta : ndarray, shape (s,)
      external step size
    
    epsilon : float
      precision for binary search
     
    simplified : booleans
      plot the simplified Lyapunov
    
    opt : boolean
      plot convergence
    
    contraction : boolean
      plot contraction
    
    Returns
    -------
    
    """
    # create empty lists
    ratesB = np.zeros(len(gammas))
    ratesG = np.zeros(len(gammas))
    ratesGspl = np.zeros(len(gammas))
    for i in range(len(gammas)):
        print(gammas[i])
        if opt == True : 
            ratesB[i] = LMM_convergence(L, mu, gammas[i], alpha, beta, epsilon)
        if contraction == True : 
            if simplified == True : 
                ratesG[i] = LMM_contraction_simplified(L, mu, gammas[i], alpha, beta, epsilon)
            ratesGspl[i] = LMM_contraction_simplified(L, mu, gammas[i], alpha, beta, epsilon)
    if opt == True:
        plt.plot(gammas, ratesB, '-x', label='convergence lyapunov')
    if contraction == True:
        if simplified == True :
            plt.plot(gammas, ratesG, '-x', label='contraction distances')
        plt.plot(gammas, ratesGspl, '-x', label='contraction distances simplified')
    
    plt.legend()
    plt.xlabel('step size')
    plt.ylabel('rates')
    plt.show()


def plot_LMM_condition_number(L : float, mus : np.ndarray, gamma : float, alpha, beta, epsilon=10**(-3), simplified = True, opt=True, contraction = True):
    """
    Compute contraction and convergence rates in terms of B and G-stability for a Runge-Kutta scheme
    as a function of the condition number L/mus
    
    Parameters
    ----------
    L : float
      smoothness parameter for the class of functions
    
    mus : np.ndarray, shape (n,)
      strong-convexity parameter for the class of functions
    
    gamma : float
      step size of the method
    
    alpha : ndarray, shape (s,)
      internal step size
    
    beta : ndarray, shape (s,)
      external step size
    
    epsilon : float
      precision for binary search
     
    simplified : booleans
      plot the simplified Lyapunov
    
    opt : boolean
      plot convergence
    
    contraction : boolean
      plot contraction
    
    Returns
    -------
    
    """
    # create empty lists
    ratesB = np.zeros(len(mus))
    ratesG = np.zeros(len(mus))
    ratesGspl = np.zeros(len(mus))
    for i in range(len(mus)):
        print(mus[i])
        if opt == True : 
            ratesB[i] = LMM_convergence(L, mus[i], gamma, alpha, beta, epsilon)
        if contraction == True : 
            if simplified == True : 
                ratesG[i] = LMM_contraction_simplified(L, mus[i], gamma, alpha, beta, epsilon)
            ratesGspl[i] = LMM_contraction_simplified(L, mus[i], gamma, alpha, beta, epsilon)
    if opt == True:
        plt.plot(L/mus, ratesB, '-x', label='convergence lyapunov')
    if contraction == True:
        if simplified == True :
            plt.plot(L/mus, ratesG, '-x', label='contraction distances')
        plt.plot(L/mus, ratesGspl, '-x', label='contraction distances simplified')
    
    plt.legend()
    plt.xlabel('condition number')
    plt.ylabel('rates')
    plt.semilogy()
    plt.semilogx()
    plt.show()