###############################################################################
# PLOT RK AS A FUNCTION OF THE CONDITION NUMBER AND OF THE STEP SIZE
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Import usefull functions
from RK_B_stability import B_stability, B_stability_optimality
from RK_G_stability import G_stability_RK, G_stability_RK_optimality


def plot_RK_step_size(L : float, mu : float, gammas : np.ndarray, alpha, beta, epsilon=10**(-3), B=True, G=True, opt=True, contraction = True):
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
     
    B : boolean
      plot B_stability
    
    G : boolean
      plot G_stability
    
    opt : boolean
      plot convergence
    
    contraction : boolean
      plot contraction
    
    Returns
    -------
    
    """
    # create empty lists
    ratesB = np.zeros(len(gammas))
    ratesBopt = np.zeros(len(gammas))
    ratesG = np.zeros(len(gammas))
    ratesGopt = np.zeros(len(gammas))
    for i in range(len(gammas)):
        print(gammas[i])
        if B == True : 
            if opt == True : 
                ratesBopt[i] = B_stability_optimality(L, mu, gammas[i], alpha, beta)
            if contraction == True : 
                ratesB[i] = B_stability(L, mu, gammas[i], alpha, beta)
        if G == True :
            if opt == True : 
                ratesGopt[i] = G_stability_RK_optimality(L, mu, gammas[i], alpha, beta, epsilon)
            if contraction == True :
                ratesG[i] = G_stability_RK(L, mu, gammas[i], alpha, beta, epsilon)
    if B == True : 
        if opt == True:
            plt.plot(gammas, ratesBopt, '-x', label='convergence distances')
        if contraction == True:
            plt.plot(gammas, ratesB, '-x', label='contraction distances')
    if G == True : 
        if opt == True:
            plt.plot(gammas, ratesGopt, '-x', label='convergence Lyapunov')
        if contraction == True:
            plt.plot(gammas, ratesG, '-x', label='contraction Lypuanov')
    plt.legend()
    plt.xlabel('step size')
    plt.ylabel('rates')
    plt.show()


def plot_RK_condition_number(L : float, mus : np.ndarray, gamma : float, alpha, beta, epsilon=10**(-3), B=True, G=True, opt=True, contraction = True):
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
     
    B : boolean
      plot B_stability
    
    G : boolean
      plot G_stability
    
    opt : boolean
      plot convergence
    
    contraction : boolean
      plot contraction
    
    Returns
    -------
    
    """
    # create empty lists
    ratesB = np.zeros(len(mus))
    ratesBopt = np.zeros(len(mus))
    ratesG = np.zeros(len(mus))
    ratesGopt = np.zeros(len(mus))
    for i in range(len(mus)):
        if B == True : 
            if opt == True : 
                ratesBopt[i] = B_stability_optimality(L, mus[i], gamma, alpha, beta)
            if contraction == True : 
                ratesB[i] = B_stability(L, mus[i], gamma, alpha, beta)
        if G == True :
            if opt == True : 
                ratesGopt[i] = G_stability_RK_optimality(L, mus[i], gamma, alpha, beta, epsilon)
            if contraction == True :
                ratesG[i] = G_stability_RK(L, mus[i], gamma, alpha, beta, epsilon)
    if B == True : 
        if opt == True:
            plt.plot(L/mus, ratesBopt, '-x', label='convergence distances')
        if contraction == True:
            plt.plot(L/mus, ratesB, '-x', label='contraction distances')
    if G == True : 
        if opt == True:
            plt.plot(L/mus, ratesGopt, '-x', label='convergence Lyapunov')
        if contraction == True:
            plt.plot(L/mus, ratesG, '-x', label='contraction Lypuanov')
    plt.legend()
    plt.xlabel('condition number')
    plt.ylabel('rates')
    plt.semilogy()
    plt.semilogx()
    plt.show()
                
    
    
    
    
    
    
    
    
    