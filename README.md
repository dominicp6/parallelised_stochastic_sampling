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

Initialise a stochastic dynamics

```
od_ld = OverdampedLangevin(x0=0.0, potential=double_well_potential, beta=1, time_step=5e-3)
ud_ld = UnderdampedLangevin(Q0=0.0, P0=0.0, potential=double_well_potential, M=1, T=1, gamma=1, time_step=5e-3)
gdd = GaussianDriftDiffusion(x0=0.0, potential=double_well_potential, diffusion_coeff=1.0, jump_prob=0.05, jump_amplitude=0.03, time_step=5e-3)
```

