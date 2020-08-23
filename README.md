# PEP_integration_optimization
 Internship Inria (SIERRA)

This project aims at transfering optimization techniques to integration methods, called Performance Estimation Problems (PEP).
The following code provide a numerical tool to analyse algorithms, from optimization or integration, in terms of convergence to optimality and contraction between two sequences.
We formulate the search for convergence and contraction rates over the family of smooth-stronly convex functions as a worst-case analysis that can be handled thnks to semi-definite programming. 
More detail is given in the references.

we provide the code for analyzing two families of methods from the integration setting : 
- Runge-Kutta methods (gradient descent, extragradient, implicit Euler)
- Linear-Multi-Step metods (Polyak Heavy Ball, Nesterov's accelerated method, Triple Momentum, etc.)



## Getting Started


### Prerequisites

- [Anaconda](https://www.anaconda.com/products/individual) : at least version 3.6
- cvxpy : 
```
conda install -c conda-forge cvxpy
```
- solver SCS (already installed in cvxpy)

### Installing

You can download a file, and run the file main.py, while changing some parameters.
Each file provides a file example.py in addition of main.py.

## Running the test

Having download a subfile, you need to run main.py and select your parameter.
For all files : 
- gamma is the step size of the method
- L : smoothness parameter of the class of functions
- mu : strong convexity parameter of the the clas of function (0 <= mu < L)

We divided the codes depending on the families of functions.


### RK_schemes

One-step methods. We study them through : 
- B-stability : distance between two sequences
- G-stability : Lyapunov functions
The file main_RK.py enables to compute contraction and convergence rates for a Runge-Kutta method over a family of convex functions.
Some examples are given in Examples,  especially how to plot rates as a function of the step size and the condition number.

### LMM_schemes

Multi-step methods. We study them through : 
- Lyapunov stability : convergence and contraction
- a simplified lyapunov version
The file main_LMM.py enables to compute contraction and convergence rates for a Linear-Multi-step method over a family of convex functions.
Some examples are given in Examples, especially how to plot rates as a function of the step size and the condition number.


### Sensitivity Analysis

We provide some examples of sensitivity analysis of ther performance estimation problems : 
- computing rates at different iterations
- compute rates at iteration (k+1) when restarting at iteration k at worst-case


### Worst case function

We provide a way to plot a function that satisfies the worst-case contraction rate of a method.
Two examples given for : 
- Nesterov's accelerated method
- a designed method



## Authors

* **Céline MOUCER** *
Internship at Inria, summer 2020 
Supervised by : Francis Bach and Adrien Taylor

## References

* [PEP toolbox](https://github.com/AdrienTaylor/Performance-Estimation-Toolbox) - Adrien Taylor
* [Y. Drori, M. Teboulle : Performance Estimation Problems](https://arxiv.org/abs/1206.3209)
* [A. Taylor, J. Hendrickx, F. Glineur : Smooth Strongly Convex Interpolation and Exact Worst-case Performance of First-order Methods](https://arxiv.org/abs/1502.05666)
* [D. Scieur, V. Roulet, F. Bach, A. d'Aspremont : Integration Methods and Accelerated Optimization Algorithms](https://arxiv.org/abs/1702.06751)
