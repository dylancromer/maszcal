import pytest
from pretend import stub
import numpy as np
import smolyak
import maszcal.interpolate


def describe_GaussianProcessInterpolator():

    def describe__call__():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)
            func_vals = np.ones(params.shape)
            interpolator = maszcal.interpolate.GaussianProcessInterpolator(params, func_vals)
            params_to_eval = np.linspace(0, 1, 20)
            result = interpolator(params_to_eval)
            assert np.allclose(result, 1)


def describe_RbfInterpolator():

    def describe__call__():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)
            func_vals = np.ones(10)
            interpolator = maszcal.interpolate.RbfInterpolator(params, func_vals)
            params_to_eval = np.linspace(0, 1, 20)
            result = interpolator(params_to_eval)
            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            p = np.arange(1, 5)
            params = np.stack((p, p, p, p, p, p)).T
            func_vals = np.ones(4)
            interpolator = maszcal.interpolate.RbfInterpolator(params, func_vals)


def describe_SmolyakInterpolator():

    def describe__call__():

        def it_interpolates_a_constant_correctly():
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=3, lb=np.zeros(2), ub=np.ones(2))
            func_vals = np.ones((smolyak_grid.grid.shape[0], 2))
            interpolator = maszcal.interpolate.SmolyakInterpolator(smolyak_grid, func_vals)
            p = np.linspace(0, 1, 20)
            params_to_eval = np.stack((p, p)).T
            result = interpolator(params_to_eval)
            assert np.allclose(result, 1)
