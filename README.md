# parallelised_stochastic_sampling
Implements parallelised stochastic sampling in arbitary dimensions for the following stochastic dynamics models :
1) Overdamped Langevin dynamics, 
$\overrightarrow{dX}_t = - \beta log \overrightarrow{\nabla} V(X_t)dt + \overrightarrow{dW}_t$
3) Anisotropic overdamped Langevin dynamics, $\overrightarrow{dX}_t =\left(- \mathbf{M}(\overrightarrow{X}_t)\overrightarrow{\nabla}V(X_t) + \beta^{-1} \overrightarrow{\nabla} \cdot \mathbf{M}(\overrightarrow{X}_t)\right)dt + \sqrt{2 \beta^{-1}}\mathbf{M}^{\frac{1}{2}}(\overrightarrow{X}_t) \overrightarrow{dW}_t$ 
4) Underdamped Langevin dyanamics, $\overrightarrow{dQ}_t = \overrightarrow{P}_t dt$, $\overrightarrow{dP}_t = \left(- \nabla V(\overrightarrow{Q}_t) - \gamma \overrightarrow{P}_t\right)dt + \sqrt{2M\gamma T} \overrightarrow{dW}_t$
5) Gaussian drift-diffusion dynamics with jumps $\overrightarrow{dX}_t = - \overrightarrow{\nabla} V(\overrightarrow{X}_t) + D \overrightarrow{dW}_t + \overrightarrow{J}_t(A, p)$, where $\overrightarrow{J}_t(A, p) = A \overrightarrow{N}_t$ with probability $1-p$ and $0$ otherwise and $\overrightarrow{N}_t$ is an isotropic standard multinormal random variable.

## Files
> `Langevin.py` - main file where stochastic sampling classes are defined

> `test_Langevin.py` - unit tests

> `tutorial.ipynb` - tutorial Jupyter notebook with worked examples

## Gallery 
[//]: # (## Example Usage)

[//]: # ()
[//]: # (### Stochastic Dynamics in a 1D Double-Well Potential)

[//]: # ()
[//]: # (First specify the functional form of the potential,)

[//]: # ()
[//]: # (```)

[//]: # (def double_well_potential&#40;x&#41;:)

[//]: # (    h = 2)

[//]: # (    c = 2)

[//]: # (    return -&#40;1 / 4&#41; * &#40;x ** 2&#41; * &#40;h ** 4&#41; + &#40;1 / 2&#41; * &#40;c ** 2&#41; * &#40;x ** 4&#41;)

[//]: # (```)

[//]: # ()
[//]: # (and initialise a stochastic dynamics object e.g.)

[//]: # ()
[//]: # (```)

[//]: # (od_ld = OverdampedLangevin&#40;x0=0.0, potential=double_well_potential, beta=1, time_step=5e-3&#41;)

[//]: # (ud_ld = UnderdampedLangevin&#40;Q0=0.0, P0=0.0, potential=double_well_potential, M=1, T=1, gamma=1, time_step=5e-3&#41;)

[//]: # (gdd = GaussianDriftDiffusion&#40;x0=0.0, potential=double_well_potential, diffusion_coeff=1.0, jump_prob=0.05, jump_amplitude=0.03, time_step=5e-3&#41;)

[//]: # (```)

[//]: # (note that $x_0, Q_0, P_0$ specify the initial coordinates of the chains &#40;in this example all chains start at the origin&#41;.)

[//]: # ()
[//]: # (Next run the sampling using the simulate method e.g. to simulate overdamped Langevin dynamics with chains of length 10,000:)

[//]: # ()
[//]: # (```)

[//]: # (traj = od_ld.simulate&#40;length=10000&#41;)

[//]: # (```)

[//]: # (by default chains are run in parallel for each available CPU process. )

[//]: # ()
[//]: # (Display the probability density of samples by using the plot method)

[//]: # ()
[//]: # (```)

[//]: # (traj.plot&#40;&#41;)

[//]: # (```)

![image](https://user-images.githubusercontent.com/55788137/191993202-150e9dc5-dc0d-4b4b-8ece-acc5ff2b43da.png)

[//]: # ()
[//]: # (note that by default chains are burnt-in and the visualisation neglects the first $N$ iteratitions &#40;here $N=2000$&#41;.)

[//]: # ()
[//]: # (Visualise the chains trajectories using the plot_trajectory method)

[//]: # (```)

[//]: # (traj.set_burn_in&#40;0&#41;)

[//]: # (traj.plot_trajectory&#40;chains=[0,1,2]&#41;)

[//]: # (```)

[//]: # (here we choose to plot only the first three chains to avoid an overly-cluttered plot)

![image](https://user-images.githubusercontent.com/55788137/192000152-281081e5-8bad-4eda-a06f-e6cf8b718739.png)

[//]: # ()
[//]: # ()
[//]: # (### Stochastic Dynamics in a 2D Quadruple-Well Potential)

[//]: # ()
[//]: # (First specify the functional form of the potential,)

[//]: # ()
[//]: # (```)

[//]: # (def quadruple_well_potential&#40;x&#41;:)

[//]: # (    h = 2)

[//]: # (    c = 2)

[//]: # (    return &#40;-&#40;1 / 4&#41; * &#40;x[0] ** 2&#41; * &#40;h ** 4&#41; + &#40;1 / 2&#41; * &#40;c ** 2&#41; * &#40;x[0] ** 4&#41;&#41; + &#40;)

[//]: # (            -&#40;1 / 4&#41; * &#40;x[1] ** 2&#41; * &#40;h ** 4&#41; + &#40;1 / 2&#41; * &#40;c ** 2&#41; * &#40;x[1] ** 4&#41;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (and initialise a stochastic dynamics, specifying the number of spatial dimensions &#40;2&#41;, e.g. )

[//]: # ()
[//]: # (```)

[//]: # (ud_ld = UnderdampedLangevin&#40;spatial_dim = 2, muQ=[0.0, 0.0], muP=[0.0, 0.0], sigmaQ=[0.5, 0.5], sigmaP=[0.5, 0.5], potential=quadruple_well_potential, M=1, T=1, gamma=1, time_step=5e-3&#41;)

[//]: # (```)

[//]: # (note this time we have randomised the chain initial conditions with specified position and momentum mean &#40;muQ, muP&#41; and standard deviation &#40;sigmaQ, sigmaP&#41; for each dimension. )

[//]: # ()
[//]: # (Finally, run a stochastic simulation)

[//]: # (```)

[//]: # (traj = ud_ld.simulate&#40;length=50000&#41;)

[//]: # (```)

[//]: # (optionally, we can modify the burn-in)

[//]: # (```)

[//]: # (traj.set_burn_in&#40;5000&#41;)

[//]: # (```)

[//]: # (and display the probability density of each parameter for each dimension using the plot method )

[//]: # (```)

[//]: # (traj.plot&#40;&#41;)

[//]: # (```)

![image](https://user-images.githubusercontent.com/55788137/191995714-eb56d98a-8d07-4c3d-a994-013b5adaa841.png)


