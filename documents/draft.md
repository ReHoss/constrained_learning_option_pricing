- Thibeau Marie
- Longstaff Schwartz
- Bernard Lapeyre avec garanties 
- On bermudan options
- On the limitations exhibited by the paper of Zhang (n=5 but the approach goes back to n=1)
- There is a kink at tau -> 0 and a kink at s* for the bermudan problem !

TODO:
- Download references from Lapeyre paper.
- Try the ANALYTICAL Green function approach (maybe something can be done through neural operators ?!).

- Understand the kink at s* on the residual loss?


---

Some remarks:
- Learnt gamma should be smooth but is not matching at all ! --> Learning the analytical formula (he strike?)
- Idea: use the analytical integral formula ?!
- The 
- https://openaccess.city.ac.uk/id/eprint/5993/  [Methodology]





ON THE NUMERICAL INTEGRATION APPROACH:

Here is the formalized summary of the numerical integration approach versus the neural network approach:Exactness in low dimensions: Evaluating the discounted risk-neutral expectation via numerical quadrature is mathematically exact and computationally optimal for low-dimensional spaces, specifically when the dimension of the underlying asset space is $d \in \{1, 2\}$.The curse of dimensionality: For multi-asset derivatives ($d \ge 3$), the cardinality of the tensor-product grid required for numerical integration scales as $\mathcal{O}(N^d)$. This exponential complexity renders pure numerical integration intractable for high-dimensional basket or maximum options.Quadrature degradation at the boundary: High-order numerical integration schemes require the integrand to be of class $C^{\infty}$. The $C^0$ singularity at the optimal exercise boundary $s^*$ degrades the theoretical convergence rate of the quadrature unless the integral is explicitly partitioned at $s^*$.The neural operator advantage: Neural networks bypass the requirement for fixed spatial grids, allowing the computational complexity to scale polynomially with $d$.


- ON DIMENSIONALITY

You have correctly identified a critical nuance. In the paper by Zhang et al. (2026), they do not actually evaluate a general high-dimensional payoff like a 5-asset Call-on-Max; they carefully restrict their high-dimensional experiments to avoid the exact tractability wall you mentioned.Here is exactly how they handle the analytical function $g_2$ in their multi-asset experiments.1. The 5-Asset Case: The Geometric Average CollapseIn Section 4.3, the authors test an American put option on the geometric mean of up to $n=5$ assets.They do not use a true 5-dimensional formula. Because the geometric average of $n$ geometric Brownian motions is itself a geometric Brownian motion, the high-dimensional problem mathematically collapses into a one-dimensional process with a modified drift $q_I$ and volatility $\sigma_I$.Consequently, the authors can rely on the standard 1-dimensional Black-Scholes analytical solution. They define $g_2$ using the first-order Taylor expansion of this collapsed 1D European analytical solution. The network is taking $n+1$ inputs, but the analytical "anchor" $g_2$ is fundamentally 1-dimensional, perfectly bypassing the curse of dimensionality.2. The Call-on-Max Case: Limitation to 2 DimensionsIn Section 4.4, the authors study the Call-on-Max option, but they strictly limit the dimensionality to $n=2$ assets.For exactly 2 assets, closed-form analytical formulas exist (originally from Johnson and Stulz). The authors derive the European call-on-max formula with dividends, which requires $N_2(x, y, \rho)$, the cumulative distribution function of the bivariate standard normal distribution. They use this exact formula as their $g_2$.They do not scale the Call-on-Max option to $n=5$ assets, precisely because the multivariate normal CDF becomes intractable.3. Resolving the Computational BottleneckEven in just 2 dimensions, evaluating the bivariate normal distribution $N_2$ involves double integrals, making it computationally expensive for a neural network training loop. To make the training tractable, the authors implement two specific techniques:Numerical Approximation of the CDF: Instead of evaluating the true double integral, they employ numerical approximations using Gauss-Legendre integration and Taylor expansion to rapidly approximate the bivariate normal distribution.The Gradient Bypass Trick: The most expensive operation in training is calculating the second-order spatial derivatives of $g_2$ for the PDE penalty term $\mathcal{L}_{bs}$. However, because $g_2 = V^e$, it inherently satisfies the homogeneous Black-Scholes equation $\mathcal{F}(V^e) = 0$. Therefore, the authors substitute $\mathcal{F}(g_1 u_{NN})$ in place of $\mathcal{F}(g_1 u_{NN} + g_2)$ inside the loss function. The automatic differentiation engine only computes the derivatives of the neural network $u_{NN}$, completely bypassing the need to differentiate the complex analytical formula during backpropagation.If you wish to scale your own implementation to true high-dimensional non-geometric payoffs (e.g., a 5-asset arithmetic basket), you will have to depart from Zhang et al.'s methodology and utilize analytical approximations (like moment-matching or the Bjerksund-Stensland bound) to serve as the exact terminal function $g_2$.