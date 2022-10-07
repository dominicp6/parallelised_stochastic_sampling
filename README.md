# parallelised_stochastic_sampling
Implements parallelised classes for overdamped Langevin dynamics, anisotropic overdamped Langevin dynamics, underdamped Langevin dyanamics and Gaussian drift-diffusion dynamics in arbitary dimensions.

## Example Usage

### Stochastic Dynamics in a Double-Well Potential

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
and run the sampling using the simulate method e.g.

```
traj = od_ld.simulate(length=10000)
```
by default chains are run in parallel for each available CPU process. 

Display the probability density of samples by using the plot method

```
traj.plot()
```
![image](https://user-images.githubusercontent.com/55788137/191993202-150e9dc5-dc0d-4b4b-8ece-acc5ff2b43da.png)

Visualise the chains trajectories using the plot_trajectory method
```
traj.set_burn_in(0)
traj.plot_trajectories(chains=[0,1,2])
```
here we choose to plot only the first three chains to avoid an overly-cluttered plot
![image](https://user-images.githubusercontent.com/55788137/192000152-281081e5-8bad-4eda-a06f-e6cf8b718739.png)


### Stochastic Dynamics in a 2D potential (a quadruple well)

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


