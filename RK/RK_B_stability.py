###############################################################################
### CONTRACTION AND CONVERGENCE RATES FOR RUNGE KUTTA METHOD : B-STABILITY
###############################################################################


# For a given Runge-Kutta method :
# x_1 = x_0 + gamma*sum(beta_i*g(x_i))
# x_i = x_0 + gamma*sum(alpha_j*g(_j))
#
#Compute the contraction rate : ||x_1 - y_1|| <= rho ||x_0 - y_0||
#Compute the convergence rate : ||x_1- x^*|| <= nu ||x_0 - x^*||
#
# We use Performance Estimation Problems, that is semi-definite programing on a worst-case formulation of the problem.


import numpy as np
import cvxpy as cp

#########
##CONTRACTION
#########

def B_stability(L : float, mu : float , gamma : float , alpha: np.ndarray, beta : np.ndarray):
    """
    Compute the contraction in terms of distances (B-stability), for a general Runge-Kutta method
    ||x_1 - y_1|| <= \rho ||x_0 - y_0||
    
    Parameters
    ----------
    L : float
      smoothness parameter for the class of functions
    
    mu : float
      strong-convexity parameter for the class of functions
    
    gamma : float
      step size of the method
    
    alpha : ndarray, shape (s,)
      internal step size
    
    beta : ndarray, shape (s,)
      external step size
    
    Returns
    -------
    rho : float
      contraction rate
    
    """
    ## PARAMETERS
    stage = len(beta) # stages of the Runge-Kutta method
    dimG = 2 * (2 + stage) # dimension of the Gram matrix
    dimF = 2 * (1 + stage) 
    npt = 2 * (1 + stage) # number of points
    
    ## INITIALIZE
    # initial points
    x_01 = np.zeros((1, dimG))
    x_02 = np.zeros((1, dimG))
    x_01[0][0] = 1.
    x_02[0][1] = 1.
    # initial gradients
    g_01 = np.zeros((1, dimG))
    g_02 = np.zeros((1, dimG))
    g_01[0][2] = 1.
    g_02[0][3] = 1.
    # initial function values
    f_01 = np.zeros((1, dimF))
    f_02 = np.zeros((1, dimF))
    f_01[0][0] = 1.
    f_02[0][1] = 1.
    XX = [x_01, x_02]
    GG = [g_01, g_02]
    FF = [f_01, f_02]
    
    ## BASE : function values and gradients for the interpolation points
    for i in range(2, npt, 2):
        f_1 = np.zeros((1, dimF))
        f_2 = np.zeros((1, dimF))
        f_1[0][i] = 1.
        f_2[0][i + 1] = 1.
        FF.append(f_1)
        FF.append(f_2)
        g_1 = np.zeros((1, dimG))
        g_2 = np.zeros((1, dimG))
        g_1[0][i + 2] = 1.
        g_2[0][i + 3] = 1.
        GG.append(g_1)
        GG.append(g_2)
    
    ## ITERATES : construct the iterates generates by the method
    if stage > 1:
        for s in range(1, stage):
            XX.append(XX[0] - gamma * sum([alpha[s][j] * GG[2 * j] for j in range(stage)]))
            XX.append(XX[1] - gamma * sum([alpha[s][j] * GG[2 * j + 1] for j in range(stage)]))
        XX.append(XX[0] - gamma * sum([beta[j] * GG[2 * j] for j in range(stage)]))
        XX.append(XX[1] - gamma * sum([beta[j] * GG[2 * j + 1] for j in range(stage)]))
    else:
        XX.append(XX[0] - gamma * (alpha[0][0] * GG[2] + (1 - alpha[0][0]) * GG[0]))
        XX.append(XX[1] - gamma * (alpha[0][0] * GG[3] + (1 - alpha[0][0]) * GG[1]))
    
    ## PEP FORMULATION
    # VARIABLES
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable((dimF, 1))
    # CONSTRAINTS
    ### Positivity of the Gram Matrix
    constraints = [G >> 0.]
    ### Constraint on the previous iterate : ||x_0 - y_0|| <= 1.
    constraints = constraints + [(XX[0][0] - XX[1][0]) @ G @ (XX[0][0] - XX[1][0]) <= 1.]
    ### Interpolation inequalities
    for i in range(npt):
        for j in range(npt):
            if j != i:
                A = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                    1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                            mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j]) -
                                            2 * mu / L * np.dot((XX[i] - XX[j]).T, GG[i] - GG[j]))
                A = .5 * (A + A.T)
                b = FF[j] - FF[i]
                #print(b)
                constraints += [b[0] @ F[:, 0] + sum([(A @ G)[k, k] for k in range(dimG)]) <= 0.]
    # OPTIMIZE
    prob = cp.Problem(cp.Maximize((XX[-2][0] - XX[-1][0]) @ G @ (XX[-2][0] - XX[-1][0])), constraints)
    prob.solve(cp.SCS)
    
    return prob.value

#########
##CONVERGENCE
#########

def B_stability_optimality(L : float , mu : float , gamma : float , alpha : np.ndarray , beta : np.ndarray):
    """
    Compute the convergence in terms of distances (B-stability), for a general Runge-Kutta method
    ||x_1 - x^*|| <= \rho ||x_0 - x^*||
    
    Parameters
    ----------
    L : float
      smoothness parameter for the class of functions
    
    mu : float
      strong-convexity parameter for the class of functions
    
    gamma : float
      step size of the method
    
    alpha : ndarray, shape (s,)
      internal step size
    
    beta : ndarray, shape (s,)
      external step size
    
    Returns
    -------
    rho : float
      convergence rate
    
    """
    ## PARAMETERS
    stage = len(beta) # stages of the Runge-Kutta method
    dimG = (2 + stage) # dimension of the Gram matrix
    dimF = (1 + stage)
    npt = (1 + stage) + 1 # number of points
    
    ## INITIALIZE
    # initial points
    x_01 = np.zeros((1, dimG))
    x_01[0][0] = 1.
    XX = [x_01]
    
    ## BASE
    GG=[]
    FF=[]
    for i in range(dimF):
        f = np.zeros((1, dimF))
        g = np.zeros((1, dimG))
        f[0][i] = 1.
        g[0][i+1] = 1.
        FF.append(f)
        GG.append(g)
    
    ## ITERATES
    if stage > 1:
        for s in range(1, stage):
            XX.append(XX[0] - gamma * sum([alpha[s][j] * GG[j] for j in range(stage)]))
        XX.append(XX[0] - gamma * sum([beta[j] * GG[j] for j in range(stage)]))
    else:
        XX.append(XX[0] - gamma * (alpha[0][0] * GG[1] + (1 - alpha[0][0]) * GG[0]))
    XX.append(np.zeros((1, dimG)))
    GG.append(np.zeros((1, dimG)))
    FF.append(np.zeros((1, dimF)))
    
    ## PEP FORMULATION
    # VARIABLES
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable((dimF, 1))
    B = []
    # CONSTRAINTS
    ### Positivity of the Gram matrix
    constraints = [G >> 0.]
    ### Constraint on the previous iterate : ||x_0 - y_0|| <= 1.
    constraints = constraints + [(XX[0][0]) @ G @ (XX[0][0]) <= 1.]
    ### Interpolation inequalities
    for i in range(npt):
        for j in range(npt):
            if j != i:
                A = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                    1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                            mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j]) -
                                            2 * mu / L * np.dot((XX[i] - XX[j]).T, GG[i] - GG[j]))
                A = .5 * (A + A.T)
                b = FF[j] - FF[i]
                B.append(b)
                constraints += [b[0] @ F[:, 0] + sum([(A @ G)[k, k] for k in range(dimG)]) <= 0.]
    
    ## OPTIMIZE
    prob = cp.Problem(cp.Maximize((XX[-2][0]) @ G @ (XX[-2][0])), constraints)
    prob.solve(cp.SCS)
    
    return prob.value


