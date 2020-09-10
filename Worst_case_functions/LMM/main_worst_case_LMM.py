###############################################################################
### WORST-CASE FUNCTION FOR LMM METHOD
###############################################################################
# Objective : plot one of the function, and initial points satisfying worst-case scenario

#For a given linear multi-step method with two stages :
# x_{k+2} + alpha_{k+1}x_{k+1} + alpha_k x_k = gamma(beta_{k+2} g(x_{k+2}) + beta_{k+1} g(x_{k+1}) + beta_{k} g(x_{k}) )
#
#
# For Phi and V two lyapunov functions, rates are defined as follows, for X_k=[x_k, x_{k-1}, g(x_k), g(x_{k-1})] :
# Contraction rate is given by : Phi(X_k - Y_k)<= rho Phi(X_{k-1} - Y_{k-1})
#
# 1. Compute the worst-case contraction rate, the optimal Lyapunov, and the iterates (Gram matrix and function values)
# 2. Given the lyapunov at the optimum, compute the iterates and the function f while minimizing the dimension of f.
# 3. Plot the fuction evaluated at a given point
# We use Performance Estimation Problems, that is semi-definite programing on a worst-case formulation of the problem.


# import
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from worst_case_function_LMM import worst_case_LMM, LMM_contraction_F_G

## Class of functions
L = 1. # smoothness
mu = 0.1 # strong-convexity

## Parameter for binary-search
epsilon = 10**(-3)

## EXAMPLES
scheme = 'Nesterov'

if scheme == 'Nesterov':
    kappa = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
    alpha = [kappa, -(1 + kappa), 1]
    beta = [- kappa, (1 + kappa), 0]
    gamma = 1 / L
if scheme == 'TMM':
    rho = 1 - np.sqrt(mu / L)
    b = rho ** 2 / (2 - rho)
    a = rho ** 2 / ((1 + rho) * (2 - rho))
    alpha = [b, -(1 + b), 1]
    beta = [-a, (1 + a), 0]
    gamma = (1 + rho) / L
    
eps = 10**-1/2

### COMPUTE THE WORST-CASE FUNCTION IN A POINT x        
## Compute the contraction rate, and the Gram matrix/function values associated with this contraction rate
F, G, tau, N, l = LMM_contraction_F_G(L, mu, gamma, alpha, beta, epsilon)

## Suppose we have two principal dimensions : 
N = 2 

## Evaluate the function in a new point
x = np.array([1, -1])
f_, X_pts, Y_pts , G_pts, g_ = worst_case_LMM(L, mu, gamma, alpha, beta, G, F, x, eps = eps)


### PLOT THE WHOLE FUNCTION
## Plot the shape of the function, and the iterated
# parameters
nn = 21 # number of points per dimension (Nb = nn*nn)
X,Y = np.linspace(-10, 10, nn),np.linspace(-20, 5, nn) # generate a grid

# Evaluate the function on a grid
sol = np.zeros((nn,nn))
for i in range(nn):
    for j in range(nn):
        xx = np.array([X[i], Y[j]])
        sol[i][j], _, _, _, _ = worst_case_LMM(L, mu, gamma, alpha, beta, G, F, xx, eps = eps)

# Plot the worst-case iterates
X_pts=np.array(X_pts)
Y_pts = np.array(Y_pts)
X_pts1 = np.array([X_pts[2*i] for i in range(len(X_pts)//2)])
X_pts2 = np.array([X_pts[2*i+1] for i in range(len(X_pts)//2)])
Y_pts1 = np.array([Y_pts[2*i] for i in range(len(Y_pts)//2)])
Y_pts2 = np.array([Y_pts[2*i+1] for i in range(len(Y_pts)//2)])
plt.subplots(1,1,figsize=(11,7))
plt.plot(X_pts1[:,0], X_pts1[:,1], '--x', color = 'red')
plt.plot(X_pts2[:,0], X_pts2[:,1], '--x', color = 'white')
plt.plot(Y_pts1[:,0], Y_pts1[:,1], '-o', color = 'orange')
plt.plot(Y_pts2[:,0], Y_pts2[:,1], '-o', color = 'purple')
## Points from the grid
plt.pcolor(X, Y, sol.T)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

