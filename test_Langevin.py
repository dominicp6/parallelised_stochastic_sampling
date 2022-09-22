import numpy as np
from abc import abstractmethod
from unittest import TestCase

from Langevin import OverdampedLangevin, UnderdampedLangevin, GaussianDriftDiffusion, Trajectory
from potentials import double_well_potential, quadruple_well_potential


class TestTrajectory(TestCase):

    def setUp(self):
        self.test_array1 = [np.array([[1.0],[0.7],[0.5]])]
        self.test_array2 = [np.array([[1.0, -1.0],[0.7, -0.75],[0.5, -0.4]])]
        self.test_array3 = [np.array([[1.0],[0.7],[0.5]]),np.array([[1.1],[0.6],[0.4]])]
        self.test_array4 = [np.array([[1.0, -1.0],[0.7, -0.75],[0.5, -0.4]]),np.array([[1.2, -1.1],[0.6, -0.72],[0.52, -0.35]])]
        self.trajectory = Trajectory

    def test_get_trajectory(self):
        assert self.trajectory(data=self.test_array1).set_burn_in(0).get_trajectory() == self.test_array1
        assert self.trajectory(data=self.test_array2).set_burn_in(0).get_trajectory() == self.test_array2
        assert self.trajectory(data=self.test_array3).set_burn_in(0).get_trajectory() == self.test_array3
        assert self.trajectory(data=self.test_array4).set_burn_in(0).get_trajectory() == self.test_array4

    def test_get_chain(self):
        assert self.trajectory(data=self.test_array1).set_burn_in(0).get_chain(0) == np.array([[1.0],[0.7],[0.5]])
        assert self.trajectory(data=self.test_array2).set_burn_in(0).get_chain(0) == np.array([[1.0, -1.0],[0.7, -0.75],[0.5, -0.4]])
        assert self.trajectory(data=self.test_array3).set_burn_in(0).get_chain(0) == np.array([[1.0],[0.7],[0.5]])
        assert self.trajectory(data=self.test_array4).set_burn_in(0).get_chain(0) == np.array([[1.0, -1.0],[0.7, -0.75],[0.5, -0.4]])

    def test_get_dimension(self):
        assert self.trajectory(data=self.test_array1).set_burn_in(0).get_dimension(0) == [[1.0,0.7,0.5]]
        assert self.trajectory(data=self.test_array2).set_burn_in(0).get_dimension(0) == [[1.0,0.7,0.5]]
        assert self.trajectory(data=self.test_array3).set_burn_in(0).get_dimension(0) == [[1.0,0.7,0.5],[1.1,0.6,0.4]]
        assert self.trajectory(data=self.test_array4).set_burn_in(0).get_dimension(0) == [[1.0,0.7,0.5],[1.2,0.6,0.52]]

    def test_get_combined_trajectory(self):
        # TODO: finish
        print(self.trajectory(data=self.test_array1).set_burn_in(0).get_combined_trajectory(0))
        print(self.trajectory(data=self.test_array2).set_burn_in(0).get_combined_trajectory(0))
        print(self.trajectory(data=self.test_array3).set_burn_in(0).get_combined_trajectory(0))
        print(self.trajectory(data=self.test_array4).set_burn_in(0).get_combined_trajectory(0))


class BaseTestCases:
    class TestDynamicalSystem(TestCase):

        @abstractmethod
        def setUp(self):
            self.dynamical_system = None
            self.simulation_length = 100

        def _check_1D_traj(self, traj):
            # Number of chains should equal the number of processes
            assert len(traj) == self.dynamical_system._set_num_processes(parallel=True, num_processes=None)
            # Each chain should have the expected shape of a 1D trajectory
            for chain_id in range(len(traj)):
                assert traj[chain_id].shape == (self.simulation_length, 1)

        def _check_2D_traj(self, traj):
            # Number of chains should equal the number of processes
            assert len(traj) == self.dynamical_system._set_num_processes(parallel=True, num_processes=None)
            # Each chain should have the expected shape of a 2D trajectory
            for chain_id in range(len(traj)):
                assert traj[chain_id].shape == (self.simulation_length, 2)

        def test_1D_no_initialisation(self):
            if self.dynamical_system is not None:
                traj = self.dynamical_system(potential=double_well_potential).simulate(length=self.simulation_length)
                self._check_1D_traj(traj)

        def test_1D_potential(self, initialisation_dict):
            if self.dynamical_system is not None:
                traj = self.dynamical_system(potential=double_well_potential, **initialisation_dict).simulate(length=self.simulation_length)
                self._check_1D_traj(traj)

        def test_2D_no_initialisation(self):
            if self.dynamical_system is not None:
                traj = self.dynamical_system(potential=quadruple_well_potential).simulate(length=self.simulation_length)
                self._check_2D_traj(traj)

        def test_2D_potential(self, initialisation_dict):
            if self.dynamical_system is not None:
                traj = self.dynamical_system(potential=quadruple_well_potential, **initialisation_dict).simulate(length=self.simulation_length)
                self._check_2D_traj(traj)


class TestOverdampedLangevin(BaseTestCases.TestDynamicalSystem):
    def setUp(self):
        self.dynamical_system = OverdampedLangevin


class TestUnderdampedLangevin(BaseTestCases.TestDynamicalSystem):
    def setUp(self):
        self.dynamical_system = UnderdampedLangevin


class TestGaussianDriftDiffusion(BaseTestCases.TestDynamicalSystem):
    def setUp(self):
        self.dynamical_system = GaussianDriftDiffusion




