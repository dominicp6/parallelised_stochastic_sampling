#!/usr/bin/env python3
"""Runs Langevin dynamics sampling in parallel

Usage:

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

    def __init__(self, data: list[np.array]):
        self.raw_data = data
        self.burn_in = min(int(data[0].shape[0]/2), 2000)  # TODO: tidy
        self.num_chains = len(data)
        assert len(set(data[i].shape[1] for i in range(len(data)))) == 1, "not all chains have the same dimension"
        self.dimension = data[0].shape[1]
        self.chain_lengths = [chain.shape[0] for chain in data]
        self.num_iterations = data[0].shape[0]

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def get_trajectory(self):
        return [self.raw_data[chain_id][self.burn_in:, :] for chain_id in range(self.num_chains)]

    def get_chain(self, chain_id: int):
        assert 0 <= chain_id < self.num_chains, f"chain_id must be an integer in the range [{0, self.num_chains})"
        return self.raw_data[chain_id][self.burn_in:]

    def get_dimension(self, dimension_id: int, chains=None):
        assert 0 <= dimension_id < self.num_chains, f"dimension_id must be an integer in the range [{0, self.dimension})"
        if chains is None:
            chains = range(self.num_chains)
        return [self.raw_data[chain_id][self.burn_in:, dimension_id] for chain_id in chains]

    def get_combined_trajectory(self):
        return np.concatenate(self.get_trajectory()).ravel()

    def plot(self, bins=100, stacked=True, chains=None):
        fig, ax = plt.subplots(nrows=1, ncols=self.dimension)
        for dim in range(self.dimension):
            if self.dimension > 1:
                axis = ax[dim]
            else:
                axis = ax
            axis.hist(self.get_dimension(dim, chains), bins, density=True, histtype='bar', stacked=stacked)
            axis.set_xlabel(f'dimension {dim}')

        fig.suptitle(f'#chains {self.num_chains}, #iter/chain {self.num_iterations}, #burn_in {self.burn_in}')
        plt.show()

        return fig, ax


class DynamicalSystem:

    def __init__(self, dimension, time_step):
        self.dim = dimension
        assert time_step > 0, "time_step must be positive"
        self.time_step = time_step
        self.x = None
        self.initial_states = None

    def simulate(self, length, seed=0, parallel=True, num_processes=None):
        num_processes = self._set_num_processes(parallel, num_processes)
        print(f"Running {num_processes} chains in parallel of length {length}.")
        self.initial_states = self._initialise(num_processes)
        print(self.initial_states)

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
        trajectory = np.zeros((length, self.dim))
        if process_id == 0:
            for idx, step in enumerate(tqdm(range(length))):
                if idx == 0:
                    self.x = self.initial_states[process_id]
                self.step()
                trajectory[step, :] = self.x
        else:
            for idx, step in enumerate(range(length)):
                if idx == 0:
                    self.x = self.initial_states[process_id]
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
                assert (len(obj) == self.dim), f"dimension is {self.dim} but {name} was dimension {len(obj)}"
            else:
                raise ValueError(f"{name} must be either float or list[float]")

    def _set_initial_points(self, x0, mu, sigma, num_processes):
        # Initialises starting coordinates for trajectories
        if x0 is not None:
            # Using specified initial coordinates
            if isinstance(x0, float):
                print('TEST 1')
                test = np.full((num_processes, self.dim), x0)
                assert test.shape == (
                num_processes, self.dim), f"shape is {test.shape} but should be {(self.dim, num_processes)}"
                return test  # TODO: fix bug if x0 = [0.0] for example
            else:
                print('TEST 2')
                test = np.tile(x0, (num_processes, 1))
                assert test.shape == (
                num_processes, self.dim), f"shape is {test.shape} but should be {(self.dim, num_processes)}"
                return test  # TODO:check
        else:
            # Using random initialisation based on specified mean(s) and std(s)
            if isinstance(mu, float) and isinstance(sigma, float):
                print('TEST 3')
                test = np.random.normal(mu, sigma, size=(num_processes, self.dim))
                assert test.shape == (
                num_processes, self.dim), f"shape is {test.shape} but should be {(self.dim, num_processes)}"
                print(test.shape)
                return test
            else:
                # mu and sigma are lists
                print('TEST 4')
                test = np.random.multivariate_normal(mu, np.diag(sigma), size=num_processes)
                assert test.shape == (
                num_processes, self.dim), f"shape is {test.shape} but should be {(self.dim, num_processes)}"
                return test  # TODO:check

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

    def __init__(self, potential, beta, time_step, x0=None, mu=None, sigma=None, dimension=1):
        super().__init__(dimension, time_step)
        assert x0 is not None or (mu is not None and sigma is not None), \
            "ERROR: specify the initial coordinate (x0) or the mean (mu) and standard deviation " \
            "(sigma) for a random initialisation."

        self._verify_obj_type_and_dim(name='x0', obj=x0)
        self._verify_obj_type_and_dim(name='mu', obj=mu)
        self._verify_obj_type_and_dim(name='sigma', obj=sigma)
        self.x0, self.mu, self.sigma = x0, mu, sigma

        log_probability = lambda x: - beta * potential(x)
        if self.dim == 1:
            self.grad_log_prob = grad(log_probability)
        else:
            self.grad_log_prob = lambda x: np.array([float(array) for array in grad(log_probability)(x)])

    def _initialise(self, num_processes):
        return self._set_initial_points(self.x0, self.mu, self.sigma, num_processes)

    def step(self):
        # Euler-Maruyama step of Overdamped Langevin dynamics
        w = np.random.normal(size=self.dim)
        self.x += self.time_step * self.grad_log_prob(self.x) + np.sqrt(2 * self.dim * self.time_step) * w


class UnderdampedLangevin(DynamicalSystem):

    def __init__(self, potential, gamma, M, T, time_step,
                 Q0=None, P0=None, muQ=None, sigmaQ=None, muP=None, sigmaP=None, dimension=1):
        super().__init__(dimension, time_step)
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

        if self.dim == 1:
            self.grad_U = grad(potential)
        else:
            self.grad_U = lambda x: np.array([float(array) for array in grad(potential)(x)])

    def _initialise(self, num_processes):
        return [self._set_initial_points(self.Q0, self.muQ, self.sigmaQ, num_processes),
                self._set_initial_points(self.P0, self.muP, self.sigmaP, num_processes)]

    def step(self):
        # Euler-Maruyama step of Underdamped Langevin dynamics
        w = np.random.normal(size=self.dim)
        self.x[0] += self.x[1] * self.time_step
        self.x[1] += (-self.grad_U(self.x[0]) - self.gamma * self.x[1]) * self.time_step \
                     + np.sqrt(2 * self.M * self.gamma * self.T * self.time_step) * w


class GaussianDriftDiffusion(DynamicalSystem):

    def __init__(self, potential, diffusion_coeff, time_step, jump_prob=0, jump_amplitude=1,
                 x0=None, mu=None, sigma=None, dimension=1):
        super().__init__(dimension, time_step)
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
        return self._set_initial_points(self.x0, self.mu, self.sigma, num_processes)

    def step(self):
        # drift-diffusion dynamics
        self.x += self.drift_force(self.x) * self.time_step + \
                  np.random.normal(scale=self.diffusion_coeff) * np.sqrt(self.time_step)
        # discrete jumps
        w = np.random.uniform(0, 1)
        if w < self.jump_prob:
            self.x += np.random.normal(scale=self.jump_amplitude)


# TODO: MetropolisHastings
# class MetropolisHastings(DynamicalSystem):

# TODO: MALA
# class MetropolisHastings(DynamicalSystem):


if __name__ == "__main__":
    from potentials import double_well_potential, quadruple_well_potential

    temperature = 30
    R = 0.0083144621  # Universal Gas Constant kJ/K/mol
    beta = 1.0 / (temperature * R)  # units (kJ/mol)**(-1)

    # Overdamped 1D

    # TEST 1.1
    # ld = OverdampedLangevin(x0=1.0, potential=double_well_potential, beta=beta, time_step=5e-3)
    # traj = ld.simulate(length=10000)
    # traj.plot()

    # TEST 1.2
    # ld = OverdampedLangevin(x0=[1.0], potential=double_well_potential, beta=beta, time_step=5e-3)
    # traj = ld.simulate(length=10000)
    # traj.plot()

    # TEST 2
    # ld = OverdampedLangevin(x0=[1.0,-1.0], potential=quadruple_well_potential, beta=beta, time_step=5e-3, dimension=2)
    # traj = ld.simulate(length=10000)
    # traj.plot()

    # TEST 3
    # ld = OverdampedLangevin(mu=0.0, sigma=1.0, potential=double_well_potential, beta=beta, time_step=5e-3)
    # traj = ld.simulate(length=50000)
    # traj.plot()

    # TEST 4.1
    # ld = OverdampedLangevin(mu=0.0, sigma=1.0, potential=quadruple_well_potential, beta=beta, time_step=5e-3, dimension=2)
    # traj = ld.simulate(length=10000)
    # traj.plot()

    # TEST 4.2
    # ld = OverdampedLangevin(x0=[-1.0, 1.0], potential=quadruple_well_potential, beta=beta, time_step=5e-3, dimension=2)
    # traj = ld.simulate(length=10000)
    # traj.plot()

    # Overdamped ND

    # Underdamped 1D

    # Underdamped ND

    # GaussianDriftDiffusionProcess 1D

    # GaussianDriftDiffusionProcess ND
