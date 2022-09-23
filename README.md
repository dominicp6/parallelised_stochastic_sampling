# stochastic_sampling
Implements parallelised classes for overdamped Langevin dynamics, underdamped Langevin dyanamics and Gaussian drift-diffusion dynamics in arbitary dimensions.

## Example Usage

### Stochastic Dynamics in a Double-Well Potential

First specify the functional form of the potential,

```
def double_well_potential(x):
    h = 2
    c = 2
    return -(1 / 4) * (x ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x ** 4)
```

and initialise a stochastic dynamics e.g.

```
od_ld = OverdampedLangevin(x0=0.0, potential=double_well_potential, beta=1, time_step=5e-3)
ud_ld = UnderdampedLangevin(Q0=0.0, P0=0.0, potential=double_well_potential, M=1, T=1, gamma=1, time_step=5e-3)
gdd = GaussianDriftDiffusion(x0=0.0, potential=double_well_potential, diffusion_coeff=1.0, jump_prob=0.05, jump_amplitude=0.03, time_step=5e-3)
```
and run the sampling using the simulate method e.g.

```
traj = ud_ld.simulate(length=10000)
```
by default chains are run in parallel for each available CPU process. 

Display the probability density using the trajectory plot method

```
traj.plot()
```
![image](https://user-images.githubusercontent.com/55788137/191993202-150e9dc5-dc0d-4b4b-8ece-acc5ff2b43da.png)


