#!/usr/bin/env python3
"""Implements parallelised classes for overdamped Langevin dynamics, underdamped Langevin dyanamics
   and Gaussian drift-diffusion dynamics in arbitary dimensions.

Example Usage:
    see https://github.com/dominicp6/stochastic_sampling/blob/master/README.md

Author:
    Dominic Phillips - 14/10/2022

"""

import numpy as np
from autograd import grad
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, cpu_count
from abc import abstractmethod


class Trajectory:

    def __init__(self, data: list[np.array], burn_in=None):
        self.raw_data = data
        if burn_in is None:
            self.burn_in = min(int(data[0].shape[0]/2), 2000) # TODO: check
        else:
            self.burn_in = burn_in
        self.num_chains = len(data)
        assert len(set(data[i].shape[1] for i in range(len(data)))) == 1, "not all chains have the same dimension"
        self.spatial_dim = data[0][0].shape[1]
        self.parameter_dim = data[0][0].shape[0]
        self.chain_lengths = [chain.shape[0] for chain in data]
        assert len(set(self.chain_lengths)) == 1, "not all chains have the same length"
        self.num_iterations = data[0].shape[0]

    def __str__(self):
        return f"Trajectory_Object(num_chains={self.num_chains}, chain_lengths={self.chain_lengths}, spatial_dim={self.spatial_dim}, parameter_dim={self.parameter_dim})"

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def get_trajectory(self):
        return [self.raw_data[chain_id][self.burn_in:, :] for chain_id in range(self.num_chains)]

    def get_chain(self, chain_id: int):
        assert 0 <= chain_id < self.num_chains, f"chain_id must be an integer in the range [{0, self.num_chains})"
        return self.raw_data[chain_id][self.burn_in:]

    def get_dimension(self, parameter_dim_id: int, spatial_dim_id: int, chains=None):
        assert 0 <= spatial_dim_id < self.num_chains, f"spatial_dim_id must be an integer in the range [{0, self.spatial_dim})"
        assert 0 <= parameter_dim_id < self.num_chains, f"parameter_dim_id must be an integer in the range [{0, self.parameter_dim})"
        if chains is None:
            chains = range(self.num_chains)
        print([self.raw_data[chain_id][self.burn_in:, parameter_dim_id, spatial_dim_id] for chain_id in chains])
        return [self.raw_data[chain_id][self.burn_in:, parameter_dim_id, spatial_dim_id] for chain_id in chains]

    def get_combined_trajectory(self):
        return np.concatenate(self.get_trajectory()).ravel()

    def plot(self, bins=100, stacked=True, chains=None):
        fig, ax = plt.subplots(nrows=self.parameter_dim, ncols=self.spatial_dim)
        for s_dim in range(self.spatial_dim):
            for p_dim in range(self.parameter_dim):
                if self.spatial_dim > 1 and self.parameter_dim > 1:
                    axis = ax[p_dim, s_dim]
                elif self.spatial_dim > 1 and self.parameter_dim == 1:
                    axis = ax[s_dim]
                elif self.spatial_dim == 1 and self.parameter_dim > 1:
                    axis = ax[p_dim]
                else:
                    axis = ax
                axis.hist(self.get_dimension(p_dim, s_dim, chains), bins, density=True, histtype='bar', stacked=stacked)
                if p_dim == 0:
                    axis.set_title(f'dimension {s_dim}', fontsize=14)
                if s_dim == 0:
                    axis.set_ylabel(f'parameter {p_dim}', fontsize=14)

        fig.suptitle(f'#chains {self.num_chains}, #iter/chain {self.num_iterations}, #burn_in {self.burn_in}')
        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_trajectory(self, chains=None):
        if chains is None:
            chains = range(self.num_chains)
        fig, ax = plt.subplots(nrows=self.parameter_dim, ncols=self.spatial_dim)
        for s_dim in range(self.spatial_dim):
            for p_dim in range(self.parameter_dim):
                if self.spatial_dim > 1 and self.parameter_dim > 1:
                    axis = ax[p_dim, s_dim]
                elif self.spatial_dim > 1 and self.parameter_dim == 1:
                    axis = ax[s_dim]
                elif self.spatial_dim == 1 and self.parameter_dim > 1:
                    axis = ax[p_dim]
                else:
                    axis = ax
                for chain in chains:
                    axis.plot(self.get_dimension(p_dim, s_dim, [chain])[0])
                if p_dim == 0:
                    axis.set_title(f'dimension {s_dim}', fontsize=14)
                if s_dim == 0:
                    axis.set_ylabel(f'parameter {p_dim}', fontsize=14)

        fig.suptitle(f'#chains {self.num_chains}, #iter/chain {self.num_iterations}, #burn_in {self.burn_in}')
        plt.tight_layout()
        plt.show()

        return fig, ax



class DynamicalSystem:

    def __init__(self, spatial_dim, parameter_dim, time_step):
        self.spatial_dim = spatial_dim
        self.parameter_dim = parameter_dim
        assert time_step > 0, "time_step must be positive"
        self.time_step = time_step
        self.x = None
        self.initial_states = None

    def simulate(self, length, seed=0, parallel=True, num_processes=None):
        num_processes = self._set_num_processes(parallel, num_processes)
        print(f"Running {num_processes} chains in parallel of length {length}.")
        self.initial_states = self._initialise(num_processes)

        if num_processes > 1:
            # Multiprocessing trajectories
            with Manager() as manager:
                trajectories = manager.list()
                processes = []
                for process_id in range(num_processes):
                    np.random.seed(process_id + seed * num_processes)
                    p = Process(target=self._run, args=(trajectories, length, process_id))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                trajectories = list(trajectories)
        else:
            # Single process trajectories
            trajectories = []
            self._run(trajectories, length, 0)

        return Trajectory(trajectories)

    def _run(self, trajectory_array, length, process_id):
        trajectory = np.zeros((length, self.parameter_dim, self.spatial_dim))
        if process_id == 0:
            for idx, step in enumerate(tqdm(range(length))):
                if idx == 0:
                    # shape parameter_dim * num_chains * spatial dim
                    self.x = self.initial_states[:, process_id, :]
                self.step()
                trajectory[step, :] = self.x
        else:
            for idx, step in enumerate(range(length)):
                if idx == 0:
                    self.x = self.initial_states[:, process_id, :]
                self.step()
                trajectory[step, :] = self.x

        trajectory_array.append(trajectory)

    def _verify_obj_type_and_dim(self, name, obj):
        # Checks that obj has type and size compatible with dim
        if obj is not None:
            if isinstance(obj, float):
                pass
            elif isinstance(obj, list):
                assert isinstance(obj[0], float), f"{name} must be either of type float or list[float]"
                assert (len(obj) == self.spatial_dim), f"dimension is {self.spatial_dim} but {name} was dimension {len(obj)}"
            else:
                raise ValueError(f"{name} must be either float or list[float]")

    def _set_initial_points(self, x0, mu, sigma, num_processes):
        # Initialises starting coordinates for trajectories
        if x0 is not None:
            # Using specified initial coordinates
            if isinstance(x0, float):
                return np.full((num_processes, self.spatial_dim), x0)
            else:
                return np.tile(x0, (num_processes, 1))
        else:
            # Using random initialisation based on specified mean(s) and std(s)
            if isinstance(mu, float) and isinstance(sigma, float):
                return np.random.normal(mu, sigma, size=(num_processes, self.spatial_dim))
            else:
                # mu and sigma are lists
                return np.random.multivariate_normal(mu, np.diag(sigma), size=num_processes)

    @staticmethod
    def _set_num_processes(parallel, num_processes):
        # Computes number of CPU processes to use
        if parallel:
            if num_processes is None:
                num_processes = max(1, cpu_count() - 1)
            else:
                assert isinstance(num_processes, int)
        else:
            num_processes = 1

        return num_processes

    @abstractmethod
    def _initialise(self, num_processes):
        pass

    @abstractmethod
    def step(self):
        pass


class OverdampedLangevin(DynamicalSystem):

    def __init__(self, potential, beta, time_step, x0=None, mu=None, sigma=None, spatial_dim=1):
        super().__init__(spatial_dim=spatial_dim, parameter_dim=1, time_step=time_step)
        assert x0 is not None or (mu is not None and sigma is not None), \
            "ERROR: specify the initial coordinate (x0) or the mean (mu) and standard deviation " \
            "(sigma) for a random initialisation."

        self._verify_obj_type_and_dim(name='x0', obj=x0)
        self._verify_obj_type_and_dim(name='mu', obj=mu)
        self._verify_obj_type_and_dim(name='sigma', obj=sigma)
        self.x0, self.mu, self.sigma = x0, mu, sigma

        log_probability = lambda x: - beta * potential(x)
        if self.spatial_dim == 1:
            self.grad_log_prob = grad(log_probability)
        else:
            self.grad_log_prob = lambda x: np.array([float(array) for array in grad(log_probability)(x)])

    def _initialise(self, num_processes):
        return np.array([self._set_initial_points(self.x0, self.mu, self.sigma, num_processes)])

    def step(self):
        # Euler-Maruyama step of Overdamped Langevin dynamics
        w = np.random.normal(size=self.spatial_dim)
        self.x[0] += self.time_step * self.grad_log_prob(self.x[0]) + np.sqrt(2 * self.spatial_dim * self.time_step) * w


class UnderdampedLangevin(DynamicalSystem):

    def __init__(self, potential, gamma, M, T, time_step,
                 Q0=None, P0=None, muQ=None, sigmaQ=None, muP=None, sigmaP=None, spatial_dim=1):
        super().__init__(spatial_dim=spatial_dim, parameter_dim=2, time_step=time_step)
        assert Q0 is not None or (muQ is not None and sigmaQ is not None), \
            "ERROR: specify the initial coordinate (Q0) or the mean (muQ) and standard deviation " \
            "(sigmaQ) for a random initialisation."
        assert (P0 is None and muP is None and sigmaP is None) or \
               (P0 is not None or (muP is not None and sigmaP is not None)), \
            "ERROR: specify the initial coordinate (P0) or the mean (muP) and standard deviation " \
            "(sigmaP) for a random initialisation. " \
            "Alternatively, leave P0, muP and sigmaP blank for zero initial momentum."

        self._verify_obj_type_and_dim(name='Q0', obj=Q0)
        self._verify_obj_type_and_dim(name='P0', obj=P0)
        self._verify_obj_type_and_dim(name='muQ', obj=muQ)
        self._verify_obj_type_and_dim(name='sigmaQ', obj=sigmaQ)
        self._verify_obj_type_and_dim(name='muP', obj=muP)
        self._verify_obj_type_and_dim(name='sigmaP', obj=sigmaP)
        self.Q0, self.muQ, self.sigmaQ, self.P0, self.muP, self.sigmaP = Q0, muQ, sigmaQ, P0, muP, sigmaP
        self.U = potential
        self.gamma = gamma
        self.M = M
        self.T = T

        if self.spatial_dim == 1:
            self.grad_U = grad(potential)
        else:
            self.grad_U = lambda x: np.array([float(array) for array in grad(potential)(x)])

    def _initialise(self, num_processes):
        Q_array = self._set_initial_points(self.Q0, self.muQ, self.sigmaQ, num_processes)
        P_array = self._set_initial_points(self.P0, self.muP, self.sigmaP, num_processes)
        inital = np.array([Q_array, P_array])
        return inital

    def step(self):
        # Euler-Maruyama step of Underdamped Langevin dynamics
        w = np.random.normal(size=self.spatial_dim)
        self.x[0] += self.x[1] * self.time_step
        self.x[1] += (-self.grad_U(self.x[0]) - self.gamma * self.x[1]) * self.time_step \
                     + np.sqrt(2 * self.M * self.gamma * self.T * self.time_step) * w


class GaussianDriftDiffusion(DynamicalSystem):

    def __init__(self, potential, diffusion_coeff, time_step, jump_prob=0.0, jump_amplitude=1.0,
                 x0=None, mu=None, sigma=None, spatial_dim=1):
        super().__init__(spatial_dim=spatial_dim, parameter_dim=1, time_step=time_step)
        assert x0 is not None or (mu is not None and sigma is not None), \
            "ERROR: specify the initial coordinate (x0) or the mean (mu) and standard deviation " \
            "(sigma) for a random initialisation."
        assert 0 <= jump_prob < 1, "jump_prob must be in the range [0,1)"
        assert jump_amplitude > 0, "jump_amplitude must be positive"

        self._verify_obj_type_and_dim(name='x0', obj=x0)
        self._verify_obj_type_and_dim(name='mu', obj=mu)
        self._verify_obj_type_and_dim(name='sigma', obj=sigma)
        self.x0, self.mu, self.sigma = x0, mu, sigma

        self.drift_force = grad(potential)
        self.diffusion_coeff = diffusion_coeff
        self.jump_prob = jump_prob
        self.jump_amplitude = jump_amplitude
        self.time_step = time_step

    def _initialise(self, num_processes):
        return np.array([self._set_initial_points(self.x0, self.mu, self.sigma, num_processes)])

    def step(self):
        # drift-diffusion dynamics
        # NB: -ve gradient drift
        self.x[0] -= self.drift_force(self.x[0]) * self.time_step + \
                  np.random.normal(scale=self.diffusion_coeff) * np.sqrt(self.time_step)
        # discrete jumps
        w = np.random.uniform(0, 1)
        if w < self.jump_prob:
            self.x[0] += np.random.normal(scale=self.jump_amplitude)


# TODO: MetropolisHastings
# class MetropolisHastings(DynamicalSystem):

# TODO: MALA
# class MetropolisHastings(DynamicalSystem):


if __name__ == "__main__":
    from test_Langevin import double_well_potential, quadruple_well_potential

    # Dynamics in a 1D potential

    od_ld = OverdampedLangevin(x0=0.0, potential=double_well_potential, beta=1, time_step=5e-3)
    ud_ld = UnderdampedLangevin(Q0=0.0, P0=0.0, potential=double_well_potential, M=1, T=1, gamma=1, time_step=5e-3)
    gdd = GaussianDriftDiffusion(x0=0.0, potential=double_well_potential, diffusion_coeff=1.0, jump_prob=0.05,
                                 jump_amplitude=0.03, time_step=5e-3)

    traj = od_ld.simulate(length=1000)

    traj.plot()
    traj.set_burn_in(0)
    traj.plot_trajectory(chains=[0,1,2])

    # Dynamics in a 2D potential

    ud_ld = UnderdampedLangevin(spatial_dim=2, muQ=[0.0, 0.0], muP=[0.0, 0.0], sigmaQ=[0.5, 0.5], sigmaP=[0.5, 0.5],
                                potential=quadruple_well_potential, M=1, T=1, gamma=1, time_step=5e-3)

    traj = ud_ld.simulate(length=50000)
    traj.set_burn_in(5000)
    traj.plot()
