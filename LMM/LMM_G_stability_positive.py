###############################################################################
### CONTRACTION AND CONVERGENCE RATES FOR LINEAR MULTI-STEP METHOD : B-STABILITY
### POSITIVE DEFINITE LYAPUNOV
###############################################################################


# For a given linear multi-step method with two stages :
# x_{k+2} + alpha_{k+1}x_{k+1} + alpha_k x_k = gamma(beta_{k+2} g(x_{k+2}) + beta_{k+1} g(x_{k+1}) + beta_{k} g(x_{k}) )
#
#
# For Phi and V two lyapunov functions, rates are defined as follows, for X_k=[x_k, x_{k-1}, g(x_k), g(x_{k-1})] :
#    - Compute the contraction rate : Phi(X_k - Y_k)<= rho Phi(X_{k-1} - Y_{k-1})
#    - Compute the convergence rate : V(X_k- X^*) <= nu V(X_{k-1} - X^*)
#
# We use Performance Estimation Problems, that is semi-definite programing on a worst-case formulation of the problem.


import numpy as np
import cvxpy as cp

#########
##CONTRACTION
#########
def LMM_contraction_positive(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, lower=0, upper=5.,  iteration = 3):
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
        # CONSTRAINTS
        ### Sign of interpolation inequalities
        constraints = [l <= 0.]
        ### Positivity of the Lyapunov function
        constraints = constraints + [N >> 0.]
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
    return tau
    

def LMM_contraction_simplified_positive(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, lower=0, upper=5.,  iterations = 3):
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
    XX = []
    FF = []
    GG = []
    YY = []
    dimN = 2 # Size of the generalized norm in the Lypauov
    dimG = 4 + 2*iterations
    dimF = 2*iterations
    npt = 2*iterations # number of points
    
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
        
    ## VECTORS FOR THE LYAPUNOV
    Y_11 = np.array([XX[-1], XX[-3]])
    Y_12 = np.array([XX[-2], XX[-4]])
    Y_01 = np.array([XX[-3], XX[-5]])
    Y_02 = np.array([XX[-4], XX[-6]])
    
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
    # updates
    value_N = None
    value_l = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True) # Norm for the lyapunov
        l = cp.Variable((npt * (npt - 1), 1)) # dual values : interpolation inequalities
        # CONSTRAINTS
        ### sign of the inteprolation inequalities
        constraints = [l <= 0.]
        ### Positivity of the Lyapunov
        constraints = constraints + [N >> 0.]
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
    return tau
    


#########
##CONVERGENCE
#########
    
def LMM_convergence(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, lower = 0, upper = 5., iteration = 3):
    """
    Compute the convergence rate for a quadratic Lyapunov Phi (G-stability), for a linear multistep method with two stages
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
      convergence rate (square)
    
    """
    ## PARAMETERS
    dimN = 4 # Points in the Lyapunov
    dimP = 2
    dimG = 2 + iteration # dimension of the Gram matrix
    dimF = iteration 
    npt = 1 + iteration # number of points
    
    ## INITIALIZE
    FF = []
    GG = []
    XX = []
    YY = []
    #initial points
    x = np.zeros((1, dimG))
    x[0][0] = 1.
    y = np.zeros((1, dimG))
    y[0][1] = 1.
    XX.append(x)  # x_0
    XX.append(y)  # x_1
    # construction of the base
    for i in range(dimF):
        f = np.zeros((1, dimF))
        g = np.zeros((1, dimG))
        f[0][i] = 1.
        g[0][i + 2] = 1.
        FF.append(f)
        GG.append(g)
    FF.append(np.zeros((1, dimF)))
    GG.append(np.zeros((1, dimG)))
    
    ## ITERATES
    for i in range(2, dimG):
        YY.append(beta[1] * XX[i - 1] + beta[0] * XX[i - 2])  # LLM
        XX.append(-alpha[1] * XX[i - 1] - alpha[0] * XX[i - 2] - gamma * GG[i - 2])  # OLM
    YY.append(np.zeros((1, dimG)))
    
    ## VECTOR FOR THE LYAPUNOV
    Y_11 = np.array([XX[-2], XX[-3], GG[-2], GG[-3]])
    Y_01 = np.array([XX[-3], XX[-4], GG[-3], GG[-4]])
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
                if (i != npt - 2) & (j != npt -2):
                    A_pos.append(.5 * (Aij + Aij.T))
                    b_pos.append(FF[j] - FF[i])
    ## BINARY SEARCH
    # Updates
    value_N = None
    p_value = None
    l_value = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True) # Generalized norm for the Lyapunov
        p = cp.Variable(dimP) # Introduce function values in the Lyapunov
        l = cp.Variable((npt*(npt-1), 1)) # dual values : intrepolation inequalities
        # CONSTRAINTS
        ### Positivity of the interpolation inequalities
        constraints = [l <= 0.]
        ### Constraints for positivity of the Lyapunov
        constraints = constraints + [N >> 0.]
        constraints = constraints + [p >= 0.]
        ###  Normalization of the generalized norm that defines the lyapunov function
        constraints = constraints + [cp.trace(N) == 1.]
        ### Interpolation inequalities
        constraints = constraints + [sum([l[i][0] * A[i] for i in range(len(A))])
                                     + (Y_11)[:, 0].T @ N @ (Y_11)[:, 0]
                                     - tau * (Y_01)[:, 0].T @ N @ (Y_01)[:, 0] << 0.]
        constraints = constraints + [sum([l[i][0] * b[i][0] for i in range(len(b))])
                                     + p[0]*FF[-2][0] + p[1]*FF[-3][0]
                                     - tau * (p[0]*FF[-3][0] + p[1]*FF[-4][0]) <= 0.]
        # OPTIMIZE
        prob = cp.Problem(cp.Minimize(0.), constraints)
        prob.solve(cp.SCS)
        if prob.status == 'optimal':
            upper = tau
            value_N = N.value
            p_value = p.value
            l_value = l.value
        else:
            lower = tau
    return tau
