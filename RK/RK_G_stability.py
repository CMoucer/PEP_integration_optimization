###############################################################################
### CONTRACTION AND CONVERGENCE RATES FOR RUNGE KUTTA METHOD : B-STABILITY
###############################################################################


# For a given Runge-Kutta method :
# x_2 = x_1 + gamma*sum(beta_i*g(x_i1))
# x_i1 = x_1 + gamma*sum(alpha_j*g(x_j1))
#
# For Phi and V two lyapunov functions, rates are defined as follows :
#    - Compute the contraction rate : Phi(X_1 - Y_1)<= rho Phi(X_0 - Y_0)
#    - Compute the convergence rate : V(X_1- X^*) <= nu V(X_0 - X^*)
#
# We use Performance Estimation Problems, that is semi-definite programing on a worst-case formulation of the problem.


import numpy as np
import cvxpy as cp

#########
##CONTRACTION
#########

def G_stability_RK(L : float , mu : float , gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, upper = 5., lower = 0.):
    """
    Compute the contraction for a quadratic Lyapunov Phi (G-stability), for a general Runge-Kutta method
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
    
    Returns
    -------
    rho^2 : float
      convergence rate (square)
    
    """
    ## PARAMETERS
    stage = len(beta) # number of stages
    dimG = 2 * 3 * (stage) + 4  # Size of the Gram matrix
    dimF = 6 * (stage) + 2
    dimN = stage # dimension for
    npt = 6 * (stage) + 2 # Number of points
    ## INITIALIZE
    # Initial points
    x_01 = np.zeros((1, dimG))
    x_02 = np.zeros((1, dimG))
    x_01[0][0] = 1.
    x_02[0][1] = 1.
    XX = [x_01, x_02]
    
    ## BASE
    FF = []
    GG = []
    for i in range(npt):
        f = np.zeros((1, dimF))
        f[0][i] = 1.
        g = np.zeros((1, dimG))
        g[0][i + 2] = 1.
        FF.append(f)
        GG.append(g)
    
    ## ITERATES
    for i in range(0, 3):
        if stage > 1:
            for s in range(1, stage):  # internal points, except zero (which is exactly the last point)
                XX.append(XX[-2] - gamma * sum([alpha[s][j] * GG[2 * j + 2 * i * (stage)] for j in range(stage)]))
                XX.append(
                    XX[-2] - gamma * sum([alpha[s][j] * GG[2 * j + 1 + 2 * i * (stage)] for j in range(stage)]))
            # points of the scheme
            XX.append(XX[-4] - gamma * sum([beta[j] * GG[2 * j + 2 * i * (stage)] for j in range(stage)]))
            XX.append(XX[-4] - gamma * sum([beta[j] * GG[2 * j + 1 + 2 * i * (stage)] for j in range(stage)]))
        else:
            XX.append(XX[2 * i] - gamma * (alpha[0][0] * GG[2 * i + 2] + (1 - alpha[0][0]) * GG[2 * i]))
            XX.append(XX[2 * i + 1] - gamma * (alpha[0][0] * GG[2 * i + 3] + (1 - alpha[0][0]) * GG[2 * i + 1]))
    
    ## VECTORS CONSIDERED IN THE LYAPUNOV
    Y_11 = [XX[-1], GG[-1]]
    Y_12 = [XX[-2], GG[-2]]
    Y_01 = [XX[-stage * 2 - 1], GG[-stage * 2 - 1]]
    Y_02 = [XX[-stage * 2 - 2], GG[-stage * 2 - 2]]
    for i in range(1, stage):
        Y_11.append(GG[-1 - 2 * i])
        Y_12.append(GG[-2 - 2 * i])
        Y_01.append(GG[-stage * 2 - 2 * i - 1])
        Y_02.append(GG[-2 * stage - 2 - 2 * i])
    Y_11 = np.array(Y_11)
    Y_12 = np.array(Y_12)
    Y_01 = np.array(Y_01)
    Y_02 = np.array(Y_02)
    dimN = len(Y_01)
    
    ## INTERPOLATION INEQUALITIES
    A = []
    b = []
    A_pos = []
    b_pos = []
    for i in range(npt):
        for j in range(npt):
            if j != i:
                Aij = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                      1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                              mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j]) -
                                              2 * mu / L * np.dot((XX[i] - XX[j]).T, GG[i] - GG[j]))
                A.append(.5 * (Aij + Aij.T))
                b.append(FF[j] - FF[i])
                if (i <= npt - 2*stage) & (j <= npt - 2*stage):
                    A_pos.append(.5 * (Aij + Aij.T))
                    b_pos.append(FF[j] - FF[i])
    
    ## PEP FORMULATION : DUAL
    # Binary search parameter : 
    value_N = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True) # Norm on the Lyapunov
        l = cp.Variable((npt * (npt - 1), 1)) # dual values : interpolation inequalities
        nu = cp.Variable(((len(A_pos)) * (len(A_pos) - 1), 1)) # dual values for positivity : interpolation
        # CONSTRAINTS
        ## Negative dual values for interpolation inequalities
        constraints = [l <= 0.]
        ## Positivity constraints for the Lyapunov
        constraints = constraints + [nu <= 0.]
        constraints = constraints + [(Y_11 - Y_12)[:, 0].T @ N @ (Y_11 - Y_12)[:, 0]
                                     - sum([nu[i][0] * A_pos[i] for i in range(len(A_pos))]) >> 0.]
        constraints = constraints + [- sum([nu[i][0] * b_pos[i][0] for i in range(len(b_pos))]) == 0.]
        ## Normalization of the generalized norm that defines the lyapunov function
        constraints = constraints + [cp.trace(N) == 1.]
        ## Interpolation constraints
        constraints = constraints + [sum([l[i][0] * A[i] for i in range(len(A))])
                                     + ((Y_11 - Y_12)[:, 0].T @ N @ (Y_11 - Y_12)[:, 0])
                                     - tau * ((Y_01 - Y_02)[:, 0].T @ N @ (Y_01 - Y_02)[:, 0]) << 0.]
        constraints = constraints + [sum([l[i][0] * b[i][0] for i in range(len(b))]) == 0.]
        
        ## OPTIMIZE
        prob = cp.Problem(cp.Minimize(0.), constraints)
        prob.solve(cp.SCS)
        if prob.status == 'optimal':
            upper = tau
            value_N = N.value
        else:
            lower = tau
            
    return tau


#########
##CONVERGENCE
#########

def G_stability_RK_optimality(L : float, mu : float , gamma : float , alpha : np.ndarray, beta : np.ndarray, epsilon : float, lower = 0, upper = 5.):
    """
    Compute the convergence for a quadratic Lyapunov (G-stability), for a general Runge-Kutta method
    Phi(x_1 - x^* )<= rho Phi(x_0 - x^*)
    
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
    
    Returns
    -------
    rho^2 : float
      convergence rate (square)
    
    """
    ## PARAMETERS
    stage = len(beta) # number of stages
    dimG = 3 * (stage) + 2  # dimension of the Gram Matrix
    dimF = 3 * (stage) + 1
    dimP = 1 # dimension for function values in the Lyapunov
    npt = 3 * (stage) + 2 # numer of points
    
    ## INITIALIZE
    # Initial points
    x_01 = np.zeros((1, dimG))
    x_01[0][0] = 1.
    XX = [x_01]
    
    ## BASE
    FF = []
    GG = []
    for i in range(dimF):
        f = np.zeros((1, dimF))
        f[0][i] = 1.
        g = np.zeros((1, dimG))
        g[0][i + 1] = 1.
        FF.append(f)
        GG.append(g)
    
    ## ITERATES
    for i in range(0, 3):
        if stage > 1:
            for s in range(1, stage): # internal points, except zero (which is exactly the last point)
                x = np.copy(XX[-1])
                XX.append(x - gamma * sum([alpha[s][j] * GG[j + i * stage] for j in range(stage)]))
            # points of the scheme
            XX.append(x - gamma * sum([beta[j] * GG[j + i * stage] for j in range(stage)]))
        else:
            XX.append(XX[i] - gamma * (alpha[0][0] * GG[i + 1] + (1 - alpha[0][0]) * GG[i]))
    XX.append(np.zeros((1, dimG)))
    GG.append(np.zeros((1, dimG)))
    FF.append(np.zeros((1, dimF)))
    
    ## VECTORS IN THE LYAPUNOV
    Y_11 = [XX[-2], GG[-2]]
    Y_01 = [XX[-stage -2], GG[-stage -2]]
    for i in range(1, stage):
        Y_11.append(GG[-2 - i])
        Y_01.append(GG[-stage - i - 2])
    Y_11 = np.array(Y_11)
    Y_01 = np.array(Y_01)
    dimN = len(Y_01)
    
    ## INTERPOLATION INEQUALITIES
    A = []
    b = []
    A_pos = []
    b_pos = []
    for i in range(npt):
        for j in range(npt):
            if j != i:
                Aij = np.dot((XX[i] - XX[j]).T, GG[j]) + \
                      1 / 2 / (1 - mu / L) * (1 / L * np.dot((GG[i] - GG[j]).T, GG[i] - GG[j]) +
                                              mu * np.dot((XX[i] - XX[j]).T, XX[i] - XX[j]) -
                                              2 * mu / L * np.dot((XX[i] - XX[j]).T, GG[i] - GG[j]))
                A.append(.5 * (Aij + Aij.T))
                b.append(FF[j] - FF[i])
                if (i != npt - 2) & (j != npt -2):
                    A_pos.append(.5 * (Aij + Aij.T))
                    b_pos.append(FF[j] - FF[i])
    
    ## PEP FORMULATION : dual
    # Binary search parameters
    value_N = None
    p_value = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True) # Norm on (x, g(x)) in the Lyapunov
        p = cp.Variable(dimP) # parameter for function values in the lyapunov
        l = cp.Variable((npt*(npt-1), 1)) # dual values associated with interpolation inequalities
        nu = cp.Variable(((npt-1) * (npt-2), 1)) # dual values associated with interpolation inequalities involved in the positivity

        # CONSTRAINTS 
        ## Negative dual values for interpolation inequalities 
        constraints = [l <= 0.]
        ## Positivity of the lyapunov
        constraints = constraints + [nu <= 0]
        constraints = constraints + [(Y_01)[:, 0].T @ N @ (Y_01)[:, 0]
                                     - sum([nu[i][0] * A_pos[i] for i in range(len(A_pos))]) >> 0.]
        constraints = constraints + [p[0]*FF[-2-stage][0]
                                    - sum([nu[i][0] * b_pos[i][0] for i in range(len(b_pos))]) >= 0.]
        ## Normalization of the generalized norm that defines the lyapunov function
        constraints = constraints + [cp.trace(N) == 1.]
        ## Interpolation inequalities
        constraints = constraints + [sum([l[i][0] * A[i] for i in range(len(A))])
                                     + (Y_11)[:, 0].T @ N @ (Y_11)[:, 0]
                                     - tau * (Y_01)[:, 0].T @ N @ (Y_01)[:, 0] << 0.]
        constraints = constraints + [sum([l[i][0] * b[i][0] for i in range(len(b))]) + p[0]*FF[-2][0] - tau*p[0]*FF[-2-stage][0] <= 0.]
        
        ## OPTIMIZE
        prob = cp.Problem(cp.Minimize(0.), constraints)
        prob.solve(cp.SCS)
        
        ## UPDATE the parameters
        if prob.status == 'optimal':
            upper = tau
            value_N = N.value
            p_value = p.value
        else:
            lower = tau
            
    return tau


