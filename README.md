# parallelised_stochastic_sampling
Implements parallelised stochastic sampling in arbitary dimensions for the following stochastic dynamics models :
1) Overdamped Langevin dynamics, 
$\overrightarrow{dX}_t = - \beta log \overrightarrow{\nabla} V(X_t)dt + \overrightarrow{dW}_t$
3) Anisotropic overdamped Langevin dynamics, $\overrightarrow{dX}_t =\left(- \mathbf{M}(\overrightarrow{X}_t)\overrightarrow{\nabla}V(X_t) + \beta^{-1} \overrightarrow{\nabla} \cdot \mathbf{M}(\overrightarrow{X}_t)\right)dt + \sqrt{2 \beta^{-1}}\mathbf{M}^{\frac{1}{2}}(\overrightarrow{X}_t) \overrightarrow{dW}_t$ 
4) Underdamped Langevin dyanamics, $\overrightarrow{dQ}_t = \overrightarrow{P}_t dt$, $\overrightarrow{dP}_t = \left(- \nabla V(\overrightarrow{Q}_t) - \gamma \overrightarrow{P}_t\right)dt + \sqrt{2M\gamma T} \overrightarrow{dW}_t$
5) Gaussian drift-diffusion dynamics with jumps $\overrightarrow{dX}_t = - \overrightarrow{\nabla} V(\overrightarrow{X}_t) + D \overrightarrow{dW}_t + \overrightarrow{J}_t(A, p)$, where $\overrightarrow{J}_t(A, p) = A \overrightarrow{N}_t$ with probability $1-p$ and $0$ otherwise and $\overrightarrow{N}_t$ is an isotropic standard multinormal random variable.

## Example Usage

### Stochastic Dynamics in a 1D Double-Well Potential

First specify the functional form of the potential,

```
def double_well_potential(x):
    h = 2
    c = 2
    return -(1 / 4) * (x ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x ** 4)
```

and initialise a stochastic dynamics object e.g.

```
od_ld = OverdampedLangevin(x0=0.0, potential=double_well_potential, beta=1, time_step=5e-3)
ud_ld = UnderdampedLangevin(Q0=0.0, P0=0.0, potential=double_well_potential, M=1, T=1, gamma=1, time_step=5e-3)
gdd = GaussianDriftDiffusion(x0=0.0, potential=double_well_potential, diffusion_coeff=1.0, jump_prob=0.05, jump_amplitude=0.03, time_step=5e-3)
```
note that $x_0, Q_0, P_0$ specify the initial coordinates of the chains (in this example all chains start at the origin).

Next run the sampling using the simulate method e.g. to simulate overdamped Langevin dynamics with chains of length 10,000:

```
traj = od_ld.simulate(length=10000)
```
by default chains are run in parallel for each available CPU process. 

Display the probability density of samples by using the plot method

```
traj.plot()
```
![image](https://user-images.githubusercontent.com/55788137/191993202-150e9dc5-dc0d-4b4b-8ece-acc5ff2b43da.png)

note that by default chains are burnt-in and the visualisation neglects the first $N$ iteratitions (here $N=2000$).

Visualise the chains trajectories using the plot_trajectory method
```
traj.set_burn_in(0)
traj.plot_trajectory(chains=[0,1,2])
```
here we choose to plot only the first three chains to avoid an overly-cluttered plot
![image](https://user-images.githubusercontent.com/55788137/192000152-281081e5-8bad-4eda-a06f-e6cf8b718739.png)


### Stochastic Dynamics in a 2D Quadruple-Well Potential

First specify the functional form of the potential,

```
def quadruple_well_potential(x):
    h = 2
    c = 2
    return (-(1 / 4) * (x[0] ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x[0] ** 4)) + (
            -(1 / 4) * (x[1] ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x[1] ** 4))
```

and initialise a stochastic dynamics, specifying the number of spatial dimensions (2), e.g. 

```
ud_ld = UnderdampedLangevin(spatial_dim = 2, muQ=[0.0, 0.0], muP=[0.0, 0.0], sigmaQ=[0.5, 0.5], sigmaP=[0.5, 0.5], potential=quadruple_well_potential, M=1, T=1, gamma=1, time_step=5e-3)
```
note this time we have randomised the chain initial conditions with specified position and momentum mean (muQ, muP) and standard deviation (sigmaQ, sigmaP) for each dimension. 

Finally, run a stochastic simulation
```
traj = ud_ld.simulate(length=50000)
```
optionally, we can modify the burn-in
```
traj.set_burn_in(5000)
```
and display the probability density of each parameter for each dimension using the plot method 
```
traj.plot()
```
![image](https://user-images.githubusercontent.com/55788137/191995714-eb56d98a-8d07-4c3d-a994-013b5adaa841.png)


