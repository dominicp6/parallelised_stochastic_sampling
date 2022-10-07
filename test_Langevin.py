import unittest

import autograd.numpy as np
from abc import abstractmethod
from unittest import TestCase

from Langevin import OverdampedLangevin, UnderdampedLangevin, GaussianDriftDiffusion, Trajectory


def double_well_potential(x):
    h = 2
    c = 2
    return -(1 / 4) * (x ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x ** 4)


def quadruple_well_potential(x):
    h = 2
    c = 2
    return (-(1 / 4) * (x[0] ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x[0] ** 4)) + (
            -(1 / 4) * (x[1] ** 2) * (h ** 4) + (1 / 2) * (c ** 2) * (x[1] ** 4))


def moro_cardin_potential(x):
    return 5 * (x[0] ** 2 - 1) ** 2 + 10 * np.arctan(7 * np.pi / 9) * x[1] ** 2


def moro_cardin_diffusion_matrix(x):
    return (1 + 8 * np.exp(- (x[0] ** 2 + x[1] ** 2) / (2 * 0.2 ** 2))) ** (-1) * np.eye((2,2))


class TestTrajectory(TestCase):

    def setUp(self):
        self.test_array1 = [np.array([np.array([[1.0], [0.7], [0.5]])])]
        self.test_array2 = [np.array([np.array([[1.0, -1.0], [0.7, -0.75], [0.5, -0.4]])])]
        self.test_array3 = [np.array([np.array([[1.0], [0.7], [0.5]])]), np.array([np.array([[1.1], [0.6], [0.4]])])]
        self.test_array4 = [np.array([np.array([[1.0, -1.0], [0.7, -0.75], [0.5, -0.4]])]),
                            np.array([np.array([[1.2, -1.1], [0.6, -0.72], [0.52, -0.35]])])]

    def test_get_trajectory(self):
        assert np.isclose(Trajectory(data=self.test_array1, burn_in=0).get_trajectory(), self.test_array1).all()
        assert np.isclose(Trajectory(data=self.test_array2, burn_in=0).get_trajectory(), self.test_array2).all()
        assert np.isclose(Trajectory(data=self.test_array3, burn_in=0).get_trajectory(), self.test_array3).all()
        assert np.isclose(Trajectory(data=self.test_array4, burn_in=0).get_trajectory(), self.test_array4).all()

    def test_get_chain(self):
        assert np.isclose(Trajectory(data=self.test_array1, burn_in=0).get_chain(0),
                          np.array([[1.0], [0.7], [0.5]])).all()
        assert np.isclose(Trajectory(data=self.test_array2, burn_in=0).get_chain(0), np.array(
            [[1.0, -1.0], [0.7, -0.75], [0.5, -0.4]])).all()
        assert np.isclose(Trajectory(data=self.test_array3, burn_in=0).get_chain(0),
                          np.array([[1.0], [0.7], [0.5]])).all()
        assert np.isclose(Trajectory(data=self.test_array4, burn_in=0).get_chain(0), np.array(
            [[1.0, -1.0], [0.7, -0.75], [0.5, -0.4]])).all()

    def test_get_dimension(self):
        assert np.isclose(Trajectory(data=self.test_array1, burn_in=0).get_dimension(0, 0), [[1.0, 0.7, 0.5]]).all()
        assert np.isclose(Trajectory(data=self.test_array2, burn_in=0).get_dimension(0, 0), [[1.0, 0.7, 0.5]]).all()
        assert np.isclose(Trajectory(data=self.test_array3, burn_in=0).get_dimension(0, 0), [[1.0, 0.7, 0.5],
                                                                                             [1.1, 0.6, 0.4]]).all()
        assert np.isclose(Trajectory(data=self.test_array4, burn_in=0).get_dimension(0, 0), [[1.0, 0.7, 0.5],
                                                                                             [1.2, 0.6, 0.52]]).all()

    def test_get_combined_trajectory(self):
        # TODO: finish
        print(Trajectory(data=self.test_array1, burn_in=0).get_combined_trajectory())
        print(Trajectory(data=self.test_array2, burn_in=0).get_combined_trajectory())
        print(Trajectory(data=self.test_array3, burn_in=0).get_combined_trajectory())
        print(Trajectory(data=self.test_array4, burn_in=0).get_combined_trajectory())


class TestDynamicalSystem():

    def initialise(self):
        self.dynamical_system = None
        self.no_initialisation_dict_1D, self.no_initialisation_dict_2D = None, None
        self.initialisation_dict_1D, self.initialisation_dict_2D = None, None
        self.random_initialisation_dict_1D, self.random_initialisation_dict_2D = None, None
        self.simulation_length = 100
        self.time_step = 5e-3
        self.parameter_dim = 1

    def _check_1D_traj(self, traj):
        # Number of chains should equal the number of processes
        assert traj.num_chains == self.dynamical_system._set_num_processes(parallel=True, num_processes=None)
        # Each chain should have the expected shape of a 1D trajectory
        for chain_id in range(traj.num_chains):
            assert traj.get_chain(chain_id).shape == (self.simulation_length, self.parameter_dim,
                                                      1), f"Chain shape {traj.get_chain(chain_id).shape} but expected {(self.simulation_length, 1)}"

    def _check_2D_traj(self, traj):
        # Number of chains should equal the number of processes
        assert traj.num_chains == self.dynamical_system._set_num_processes(parallel=True, num_processes=None)
        # Each chain should have the expected shape of a 2D trajectory
        for chain_id in range(traj.num_chains):
            assert traj.get_chain(chain_id).shape == (self.simulation_length, self.parameter_dim,
                                                      2), f"Chain shape {traj.get_chain(chain_id).shape} but expected {(self.simulation_length, 2)}"

    def test_1D_initialisation(self):
        if self.initialisation_dict_1D is not None:
            traj = self.dynamical_system(potential=double_well_potential, **self.initialisation_dict_1D).simulate(
                length=self.simulation_length)
            traj.set_burn_in(0)
            self._check_1D_traj(traj)

    def test_1D_random_initialisation(self):
        if self.random_initialisation_dict_1D is not None:
            traj = self.dynamical_system(potential=double_well_potential,
                                         **self.random_initialisation_dict_1D).simulate(length=self.simulation_length)
            traj.set_burn_in(0)
            self._check_1D_traj(traj)

    def test_2D_initialisation(self):
        if self.initialisation_dict_2D is not None:
            traj = self.dynamical_system(potential=quadruple_well_potential, **self.initialisation_dict_2D).simulate(
                length=self.simulation_length)
            traj.set_burn_in(0)
            self._check_2D_traj(traj)

    def test_2D_random_initialisation(self):
        if self.random_initialisation_dict_2D is not None:
            traj = self.dynamical_system(potential=quadruple_well_potential,
                                         **self.random_initialisation_dict_2D).simulate(length=self.simulation_length)
            traj.set_burn_in(0)
            self._check_2D_traj(traj)


class TestOverdampedLangevin(TestDynamicalSystem, TestCase):

    def setUp(self):
        super(TestOverdampedLangevin, self).initialise()
        temperature = 300
        R = 0.0083144621  # Universal Gas Constant kJ/K/mol
        beta = 1.0 / (temperature * R)  # units (kJ/mol)**(-1)
        self.dynamical_system = OverdampedLangevin
        self.no_initialisation_dict_1D = {'beta': beta, 'time_step': self.time_step}
        self.no_initialisation_dict_2D = {'beta': beta, 'time_step': self.time_step, 'spatial_dim': 2}
        self.initialisation_dict_1D = {'beta': beta, 'time_step': self.time_step, 'x0': 0.1}
        self.initialisation_dict_2D = {'beta': beta, 'time_step': self.time_step, 'x0': [0.1, -0.1], 'spatial_dim': 2}
        self.random_initialisation_dict_1D = {'beta': beta, 'time_step': self.time_step, 'mu': 0.1, 'sigma': 0.2}
        self.random_initialisation_dict_2D = {'beta': beta, 'time_step': self.time_step, 'mu': [0.1, 0.12],
                                              'sigma': [0.2, 0.1], 'spatial_dim': 2}


class TestUnderdampedLangevin(TestDynamicalSystem, TestCase):

    def setUp(self):
        super(TestUnderdampedLangevin, self).initialise()
        gamma, M, T = 1, 1, 1
        self.parameter_dim = 2
        self.dynamical_system = UnderdampedLangevin
        self.no_initialisation_dict_1D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T}
        self.no_initialisation_dict_2D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T, 'spatial_dim': 2}
        self.initialisation_dict_1D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T, 'Q0': 0.1,
                                       'P0': 0.1}
        self.initialisation_dict_2D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T, 'Q0': [0.1, -0.1],
                                       'P0': [0.1, 0.05], 'spatial_dim': 2}
        self.random_initialisation_dict_1D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T, 'muQ': 0.1,
                                              'sigmaQ': 0.05, 'muP': 0.1, 'sigmaP': 0.03}
        self.random_initialisation_dict_2D = {'gamma': gamma, 'time_step': self.time_step, 'M': M, 'T': T,
                                              'muQ': [0.1, -0.1], 'sigmaQ': [0.05, 0.02], 'muP': [0.1, 0.05],
                                              'sigmaP': [0.03, 0.06], 'spatial_dim': 2}


class TestGaussianDriftDiffusion(TestDynamicalSystem, TestCase):

    def setUp(self):
        super(TestGaussianDriftDiffusion, self).initialise()
        diff_coeff = 1
        self.dynamical_system = GaussianDriftDiffusion
        self.no_initialisation_dict_1D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step}
        self.no_initialisation_dict_2D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step, 'spatial_dim': 2}
        self.initialisation_dict_1D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step, 'jump_prob': 0.05,
                                       'jump_amplitude': 0.5, 'x0': 0.4}
        self.initialisation_dict_2D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step, 'jump_prob': 0.05,
                                       'jump_amplitude': 0.5, 'x0': [0.4, 0.2], 'spatial_dim': 2}
        self.random_initialisation_dict_1D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step,
                                              'jump_prob': 0.05,
                                              'jump_amplitude': 0.5, 'mu': 0.4, 'sigma': 0.2}
        self.random_initialisation_dict_2D = {'diffusion_coeff': diff_coeff, 'time_step': self.time_step,
                                              'jump_prob': 0.05,
                                              'jump_amplitude': 0.5, 'mu': [0.4, 0.2], 'sigma': [0.1, 0.02],
                                              'spatial_dim': 2}
