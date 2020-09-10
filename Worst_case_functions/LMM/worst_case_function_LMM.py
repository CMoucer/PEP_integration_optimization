###############################################################################
### WORST-CASE FUNCTION FOR RK METHOD
###############################################################################
# For a given linear multi-step method with two stages :
# x_{k+2} + alpha_{k+1}x_{k+1} + alpha_k x_k = gamma(beta_{k+2} g(x_{k+2}) + beta_{k+1} g(x_{k+1}) + beta_{k} g(x_{k}) )

import numpy as np
import cvxpy as cp


def LMM_contraction_F_G(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, epsilon : float, iteration = 5, lower=0., upper = 5., eps = 0.):
    """
    Compute the contraction rate for a quadratic Lyapunov Phi (G-stability), for a linear multistep method with two stages
    Phi(x_1 - y_1 )<= rho Phi(x_0 - y_0)
    Compute the associated Gram Matrix with lowest rank (trace minimization heuristic), and its function values
    
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
    
    eps : float
      tolerance for non-violating the interpolation inequalities
    
    Returns
    -------
    rho^2 : float
      contraction rate (square)
     
    G : np.ndarray , shape ((2+s)*2, (2+s)*2), 
      Gram matrix
    
    F : np.ndarray, shape ((2+s)*2, )
      function values
    
    """
    
    ## PARAMETERS
    dimN = 2 # Norm in the Lyapunov
    dimG = 4 * 2 + 2 * iteration # dimension of the Gram matrix
    dimF = 2 * 2 + 2 * iteration
    npt = 2 * 2 + 2 * iteration # number of points
    
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
                if (i <= npt - 3) & (j <= npt - 3):
                    A_pos.append(.5 * (Aij + Aij.T))
                    b_pos.append(FF[j] - FF[i])
    ## BINARY SEARCH
    # Updates
    value_N = None
    value_l = None
    while upper - lower >= epsilon:
        tau = (lower + upper) / 2
        # VARIABLES
        N = cp.Variable((dimN, dimN), symmetric=True)
        l = cp.Variable((npt * (npt - 1), 1))
        nu = cp.Variable(((npt-1)*(npt-2), 1))
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
        constraints = constraints + [sum([l[i][0] * b[i][0] for i in range(len(b))]) <= 0.]
        
        ## OPTIMIZE
        prob = cp.Problem(cp.Minimize(0.), constraints)
        prob.solve(cp.SCS)
        if prob.status == 'optimal':
            upper = tau
            value_N = N.value # lyapunov function
            value_l = l.value
        else:
            lower = tau 
    
    ## COMPUTE F,G WITH LOW RANK (trace heuristic) given the contraction rate and the lyapunov function
    # VARIBALES
    F = cp.Variable((dimF, 1)) # Function values
    G = cp.Variable((dimG, dimG), symmetric = True) # Gram matrix
    # CONSTRAINTS
    ### positivity of the Gram Matrix
    constraints = [ G >> 0.]
    ### positivity of the function (convex)
    constraints = constraints + [F >= 0.]
    ### interpolation inequalities
    for i in range(npt):
        for j in range(npt):
            if j != i:
                A = np.dot((YY[i]-YY[j]).T, GG[j]) + 1/2/(1 - mu/L)*(1/L * np.dot((GG[i]-GG[j]).T, GG[i]-GG[j]) + mu * np.dot((YY[i]-YY[j]).T, YY[i]-YY[j]) - 2*mu/L * np.dot((YY[i]-YY[j]).T, GG[i]-GG[j]))
                A = .5 * (A + A.T)
                b = FF[j]-FF[i]
                # Small violation of the interpolation constraints while reconstructing
                constraints += [b[0] @ F[:, 0] + sum([(A@G)[k, k] for k in range(dimG)]) <= -eps]
    ### Fix the value of the previous step
    constraints = constraints + [cp.trace(((Y_01 - Y_02)[:, 0].T @ value_N @ (Y_01 - Y_02)[:, 0]) @ G)  == 1.]
    ### Satisfy worst-case contraction rate
    constraints = constraints + [cp.trace(((Y_11 - Y_12)[:, 0].T @ value_N @ (Y_11 - Y_12)[:, 0]) @ G )
                                 == tau * cp.trace(((Y_01 - Y_02)[:, 0].T @ value_N @ (Y_01 - Y_02)[:, 0]) @ G)]
    
    ## OPTIMIZE
    prob = cp.Problem(cp.Minimize(cp.trace(G)), constraints)
    prob.solve(cp.SCS)
    value_G = G.value
    
    ## VERIFY THE CONTRACTION RATE
    #trace1 = np.trace(((Y_11 - Y_12)[:, 0].T @ value_N @ (Y_11 - Y_12)[:, 0]) @ value_G )
    #print(trace1)
    #trace2 = np.trace(((Y_01 - Y_02)[:, 0].T @ value_N @ (Y_01 - Y_02)[:, 0]) @ value_G)
    #print(trace2)
    #print(trace1/trace2)
    
    return F.value, G.value, tau, value_N, value_l


def worst_case_LMM(L, mu, gamma, alpha, beta, G, F, x, N=2, eps=0.):
    """
    Evaluate one of the worst-case function at a new point x
        
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
      
    G : np.ndarray , shape ((2+s)*2, (2+s)*2), 
      Gram matrix
    
    F : np.ndarray, shape ((2+s)*2, )
      function values 
    
    x : np.ndarray, shape (N,)
      values in which the function is evaluated
    
    N : int
      dimension to keep
    
    eps : float
      error to avoid violating the interpolation constraints after QR decomposition of G
      
    Returns
    -------
    f(x)
      worst-case function evaluated in x
    
    """
    ## QR DECOMPOSITION ON G, WITH N MAIN DIMENSIONS
    # Find R such that R^TR=G
    eigenvalues, eigenvector = np.linalg.eig(G)
    eigs = np.zeros(len(eigenvalues))
    for i in range(N):
        eigs[i] = eigenvalues[i]
    new_D = np.diag(np.sqrt(eigs))
    Q, R = np.linalg.qr(np.dot(new_D, eigenvector.T))
    # Keep the principal dimensions
    R = R[:N, :]
    # Reconstruction of x_i based on the approximating R
    XX=[R[:,0], R[:,1], R[:,2], R[:,3]] # initial points
    YY = []
    GG = []
    for i in range(4, R.shape[1]):
        GG.append(R[:,i])# basis
    # Iterates
    for i in range(4, R.shape[1], 2):
        YY.append(XX[i - 4] * beta[0] + beta[1] * XX[i - 2])  # y_1i
        YY.append(XX[i - 3] * beta[0] + beta[1] * XX[i - 1])  # y_2i
        XX.append(-alpha[1] * XX[i - 2] - alpha[0] * XX[i - 4] - gamma * GG[i - 4])  # x_1i
        XX.append(-alpha[1] * XX[i - 1] - alpha[0] * XX[i - 3] - gamma * GG[i - 3])  # x_2i
        
    ## INTERPOLATION OF THE FUNCTION (QCQP PROBLEM)
    m1 = np.array([[-mu*L, mu*L], [mu*L, -mu*L]])
    m2 = np.array([[mu, -L], [-mu, L]])
    m3 = np.array([[-1, 1], [1, -1]])
    M1 = np.kron(m1, np.eye(R.shape[0]))
    M2 = np.kron(m2, np.eye(R.shape[0]))
    M3 = np.kron(m3, np.eye(R.shape[0]))
    f = cp.Variable(1)
    g = cp.Variable(R.shape[0])
    constraints = [f >= 0.]
    for i in range(R.shape[1]-4):
        X = np.array([x[0], x[1], YY[i][0], YY[i][1]]).T
        G = np.array([0, 0, GG[i][0], GG[i][1]]).T
        U1 = np.zeros((2,4))
        U1[0][0] = 1.
        U1[1][1] = 1.
        G = G + g@U1
        X_ = np.array([YY[i][0], YY[i][1], x[0], x[1]]).T
        G_ = np.array([GG[i][0], GG[i][1], 0, 0]).T
        U2 = np.zeros((2,4))
        U2[0][2] = 1.
        U2[1][3] = 1.
        G_ = G_ + g@U2
        constraints = constraints + [(L-mu)*(f-F[i][0]) + 1/2 * (X.T @ M1 @ X + 2*(X.T @ M2 @ G) + cp.quad_form(G, M3)) >= -eps]
        constraints = constraints + [(L-mu)*(F[i][0]-f) + 1/2 * (X_.T @ M1 @ X_ + 2*(X_.T @ M2 @ G_) + cp.quad_form(G_, M3)) >= -eps ]
    prob = cp.Problem(cp.Minimize(f), constraints)
    prob.solve(cp.SCS)
    print('The interpolating function evaluated at point {}is equal to'.format(str(x)), prob.value)
    
    return prob.value, YY, XX, GG, g.value

