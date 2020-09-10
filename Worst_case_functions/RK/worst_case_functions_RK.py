###############################################################################
### COMPUTE WORST-CASE FUNCTION FOR RK-METHODS
###############################################################################
# For a given Runge-Kutta method :
# x_1 = x_0 + gamma*sum(beta_i*g(x_i))
# x_i = x_0 + gamma*sum(alpha_j*g(_j))


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def B_stability_RK_rate_G_F(L : float, mu : float, gamma : float , alpha : np.ndarray, beta : np.ndarray):
    """
    Compute the contraction rate for B-stability (contraction in terms of distances), the function values and its iterates
    verifying worst-case.
        
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
    rho^2 : float
      contraction rate (square)
    
    G : np.ndarray , shape ((2+s)*2, (2+s)*2), 
      Gram matrix
    
    F : np.ndarray, shape ((2+s)*2, )
      function values
    """
    ## INTERNAL PARAMETERS
    stage = len(beta)
    dimG = 2 * (2 + stage) # size of the gram matrix
    dimF = 2 * (1 + stage) # number of points at which the function is evaluated
    npt = 2 * (1 + stage) # number of points
    
    ## INITIALIZE
    # initial points
    x_01 = np.zeros((1, dimG))
    x_02 = np.zeros((1, dimG))
    x_01[0][0] = 1.
    x_02[0][1] = 1.
    XX = [x_01, x_02]
    # initial gradients
    g_01 = np.zeros((1, dimG))
    g_02 = np.zeros((1, dimG))
    g_01[0][2] = 1.
    g_02[0][3] = 1.
    GG = [g_01, g_02]
    # initial function values
    f_01 = np.zeros((1, dimF))
    f_02 = np.zeros((1, dimF))
    f_01[0][0] = 1.
    f_02[0][1] = 1.
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
    
    return prob.value, F.value, G.value


def worst_case_RK(L : float, mu : float, gamma : float, alpha : np.ndarray, beta : np.ndarray, G : np.ndarray, F : np.ndarray, x : np.ndarray, N=2, eps=0.):
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
    stage = len(beta)
    XX=[R[:,0], R[:,1]] # initial points
    GG = []
    for i in range(2, R.shape[1]):
        GG.append(R[:,i]) # basis
    # Iterates
    if stage > 1:
        for s in range(1, stage):
            XX.append(XX[0] - gamma * sum([alpha[s][j] * GG[2 * j] for j in range(stage)]))
            XX.append(XX[1] - gamma * sum([alpha[s][j] * GG[2 * j + 1] for j in range(stage)]))
        XX.append(XX[0] - gamma * sum([beta[j] * GG[2 * j] for j in range(stage)]))
        XX.append(XX[1] - gamma * sum([beta[j] * GG[2 * j + 1] for j in range(stage)]))
    else:
        XX.append(XX[0] - gamma * (alpha[0][0] * GG[2] + (1 - alpha[0][0]) * GG[0]))
        XX.append(XX[1] - gamma * (alpha[0][0] * GG[3] + (1 - alpha[0][0]) * GG[1]))
        
    ## INTERPOLATION OF THE FUNCTION (QCQP PROBLEM)
    m1 = np.array([[-mu*L, mu*L], [mu*L, -mu*L]])
    m2 = np.array([[mu, -L], [-mu, L]])
    m3 = np.array([[-1, 1], [1, -1]])
    M1 = np.kron(m1, np.eye(R.shape[0]))
    M2 = np.kron(m2, np.eye(R.shape[0]))
    M3 = np.kron(m3, np.eye(R.shape[0]))
    f = cp.Variable(1)
    g = cp.Variable(R.shape[0])
    constraints = [f >= -1.]
    for i in range(R.shape[1]-2):
        X = np.array([x[0], x[1], XX[i][0], XX[i][1]]).T
        G = np.array([0, 0, GG[i][0], GG[i][1]]).T
        U1 = np.zeros((2,4))
        U1[0][0] = 1.
        U1[1][1] = 1.
        G = G + g@U1
        X_ = np.array([XX[i][0], XX[i][1], x[0], x[1]]).T
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

    return prob.value, XX, GG, g.value





