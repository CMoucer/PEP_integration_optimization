###############################################################################
# SENSITIVITY ANALYSIS FUNCTIONS FOR LMM METHODS
###############################################################################

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def LMM_contraction_N(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, lower=0, upper=5.,  iteration = 3):
    """
    Compute the contraction rate for a quadratic Lyapunov Phi (G-stability), for a linear multistep method with two stages
    Phi(x_1 - y_1 )<= rho Phi(x_0 - y_0)
    
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
      
    epsilon : float
      precision parameter for the binary search
    
    lower : float
      lower bound for binary search
      
    upper : float
      upper bound for binary search
     
    iterations : int
      Iterations k at which contraction is observed
    
    Returns
    -------
    rho^2 : float
      contraction rate (square)
    
    """
    ## PARAMETERS
    dimN = 4 # Norm in the Lyapunov
    dimG = 4 + 2*iteration # Size of the Gram Matrix
    dimF = 2*iteration
    npt = 2*iteration # number of points
    
    ## INITIALIZE
    XX = []
    FF = []
    GG = []
    YY = []
    # Initial points
    x = np.zeros((1, dimG))
    x[0][0] = 1.
    y = np.zeros((1, dimG))
    y[0][2] = 1.
    x_ = np.zeros((1, dimG))
    x_[0][1] = 1.
    y_ = np.zeros((1, dimG))
    y_[0][3] = 1.
    XX.append(x)  # x_10
    XX.append(x_)  # x_20
    XX.append(y)  # x_11
    XX.append(y_)  # x_21
    # Construct the base
    for i in range(dimF):
        f = np.zeros((1, dimF))
        g = np.zeros((1, dimG))
        f[0][i] = 1.
        g[0][i + 4] = 1.
        FF.append(f)
        GG.append(g)
    
    ## ITERATES
    for i in range(4, dimG, 2):
        YY.append(XX[i - 4] * beta[0] + beta[1] * XX[i - 2])  # y_1i
        YY.append(XX[i - 3] * beta[0] + beta[1] * XX[i - 1])  # y_2i
        XX.append(-alpha[1] * XX[i - 2] - alpha[0] * XX[i - 4] - gamma * GG[i - 4])  # x_1i
        XX.append(-alpha[1] * XX[i - 1] - alpha[0] * XX[i - 3] - gamma * GG[i - 3])  # x_2i
    
    ## VECTOR FOR THE LYAPUNOV
    Y_11 = np.array([XX[-1], XX[-3], GG[-1], GG[-3]])
    Y_12 = np.array([XX[-2], XX[-4], GG[-2], GG[-4]])
    Y_01 = np.array([XX[-3], XX[-5], GG[-3], GG[-5]])
    Y_02 = np.array([XX[-4], XX[-6], GG[-4], GG[-6]])
    
    ## INTERPOLATION INEQUALITIES
    A = []
    b = []
    A_pos = []
    b_pos = []
    for i in range(npt):
        for j in range(npt):
            if j != i:
                Aij = np.dot((YY[i] - YY[j]).T, GG[j]) + \
                      1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                              mu * np.dot((YY[i] - YY[j]).T, YY[i] - YY[j]) -
                                              2 * mu / L * np.dot((YY[i] - YY[j]).T, GG[i] - GG[j]))
                A.append(.5 * (Aij + Aij.T))
                b.append(FF[j] - FF[i])
                if (i <= npt-3) & (j <= npt-3) :
                    A_pos.append(.5 * (Aij + Aij.T))
                    b_pos.append(FF[j] - FF[i])
    
    ## BINARY SEARCH
    # Udpates
    value_N = None
    value_l = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True) # Norm for the Lypaunov
        l = cp.Variable((npt * (npt - 1), 1)) # dual values for interpolatoin inequalities
        nu = cp.Variable(((npt-1)*(npt-2),1)) # positivity of the Lyapunov
        # CONSTRAINTS
        ### Sign of interpolation inequalities
        constraints = [l <= 0.]
        ### Positivity of the Lyapunov function
        constraints = constraints + [nu <= 0.]
        constraints = constraints + [(Y_01 - Y_02)[:, 0].T @ N @ (Y_01 - Y_02)[:, 0]
                                 - sum([nu[i][0] * A_pos[i] for i in range(len(A_pos))]) >> 0.]
        constraints = constraints + [- sum([nu[i][0] * b_pos[i][0] for i in range(len(b_pos))]) >= 0.]
        ### Normalization of the generalized norm that defines the lyapunov function
        constraints = constraints + [cp.trace(N) == 1.]
        ### Interpolation inequalities
        constraints = constraints + [sum([l[i][0] * A[i] for i in range(len(A))])
                                     + ((Y_11 - Y_12)[:, 0].T @ N @ (Y_11 - Y_12)[:, 0])
                                     - tau * ((Y_01 - Y_02)[:, 0].T @ N @ (Y_01 - Y_02)[:, 0]) << 0.]
        constraints = constraints + [sum([l[i][0] * b[i][0] for i in range(len(A))]) == 0.]
        
        ## OPTIMIZE
        prob = cp.Problem(cp.Minimize(0.), constraints)
        prob.solve(cp.SCS)
        if prob.status == 'optimal':
            upper = tau
            value_N = N.value
            value_l = l.value
        else:
            lower = tau
    return tau, value_N



def LMM_contraction_restarted(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, N_matrice : np.ndarray, last_rate : float, iteration = 7, tolerance = 0) :
    """
    Compute the contraction rate after restart for a quadratic Lyapunov Phi (G-stability), for a linear multistep method with two stages
    Phi(x_1 - y_1 )<= rho Phi(x_0 - y_0)
    
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
      
    N_matrix : np.ndarray, shape (4, 4)
      Lyapunov function at previous step
    
    last_rate : float
      Contraction rate at previous step
     
    iterations : int
      Iterations k at which contraction is observed
    
    tolerance : float
      Tolerance to verify previous contraction rate
    
    Returns
    -------
    rho^2 : float
      contraction rate (square) after restart
    
    """
    ## PARAMETERS
    dimG = 2*(iteration+2) # dimension of the Gram matrix
    dimF = 2*iteration
    npt = 2*iteration # number of points
    
    ## INITIALIZE
    XX = []
    FF = []
    GG = []
    YY = []
    # Initial points
    x = np.zeros((1, dimG))
    x[0][0] = 1.
    y = np.zeros((1, dimG))
    y[0][2] = 1.
    x_ = np.zeros((1, dimG))
    x_[0][1] = 1.
    y_ = np.zeros((1, dimG))
    y_[0][3] = 1.
    XX.append(x)  # x_10
    XX.append(x_)  # x_20
    XX.append(y)  # x_11
    XX.append(y_)  # x_21
    # Compute the base
    for i in range(dimF):
        f = np.zeros((1, dimF))
        g = np.zeros((1, dimG))
        f[0][i] = 1.
        g[0][i + 4] = 1.
        FF.append(f)
        GG.append(g)
    
    ## ITERATES
    for i in range(4, dimG, 2):
        YY.append(XX[i - 4] * beta[0] + beta[1] * XX[i - 2])  # y_1i
        YY.append(XX[i - 3] * beta[0] + beta[1] * XX[i - 1])  # y_2i
        XX.append(-alpha[1] * XX[i - 2] - alpha[0] * XX[i - 4] - gamma * GG[i - 4])  # x_1i
        XX.append(-alpha[1] * XX[i - 1] - alpha[0] * XX[i - 3] - gamma * GG[i - 3])  # x_2i
    
    ## VECTOR FOR THE LYAPUNOV
    Y_11 = np.array([XX[-1], XX[-3], GG[-1], GG[-3]])
    Y_12 = np.array([XX[-2], XX[-4], GG[-2], GG[-4]])
    Y_01 = np.array([XX[-3], XX[-5], GG[-3], GG[-5]])
    Y_02 = np.array([XX[-4], XX[-6], GG[-4], GG[-6]])
    
    ## OPTIMIZATION STEP
    # VARIABLES
    G = cp.Variable((dimG, dimG), symmetric=True) # Gram matrix
    F = cp.Variable((dimF, 1)) # Function values
    
    # CONSTRAINTS
    ### Positivity of the Gram matrix
    constraints = [G >> 0.]
    ### Restart from previous worst-case with a given tolerance
    constraints = constraints + [cp.trace(G @ ((Y_01-Y_02)[:, 0].T @ N_matrice @ (Y_01-Y_02)[:, 0])) >= last_rate - tolerance]
    constraints = constraints + [cp.trace(G @ ((Y_01-Y_02)[:, 0].T @ N_matrice @ (Y_01-Y_02)[:, 0])) <= last_rate + tolerance]
    ### interpolation inequalities
    for i in range(npt):
        for j in range(npt):
            if j != i:
                A = np.dot((YY[i]-YY[j]).T, GG[j]) + 1/2/(1 - mu/L)*(1/L * np.dot((GG[i]-GG[j]).T, GG[i]-GG[j]) 
                                                                     + mu * np.dot((YY[i]-YY[j]).T, YY[i]-YY[j]) 
                                                                     - 2*mu/L * np.dot((YY[i]-YY[j]).T, GG[i]-GG[j]))
                A = .5 * (A + A.T)
                b = FF[j]-FF[i]
                constraints += [b[0] @ F[:, 0] + sum([(A@G)[k, k] for k in range(dimG)]) <= 0.]
    
    ## OPTIMIZE
    prob = cp.Problem(cp.Maximize((cp.trace(G @ ((Y_11 - Y_12)[:, 0].T @ N_matrice @ (Y_11 - Y_12)[:, 0])))), constraints)
    prob.solve(cp.SCS)
    
    return prob.value


    
