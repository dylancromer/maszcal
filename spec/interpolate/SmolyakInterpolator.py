import pytest
import numpy as np
import smolyak
from pretend import stub
from maszcal.interpolate import SmolyakInterpolator


def describe_smolyak_interpolator():

    def describe_init():

        def correct_args_case():
            func_vals = stub()
            smolyak_grid = stub()
            SmolyakInterpolator(smolyak_grid, func_vals)

        def incorrect_args_case():
            with pytest.raises(TypeError):
                SmolyakInterpolator()

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=3, lb=np.zeros(2), ub=np.ones(2))

            func_vals = np.ones((smolyak_grid.grid.shape[0], 2))

            interpolator = SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            p = np.linspace(0, 1, 20)
            params_to_eval = np.stack((p, p)).T

            result = interpolator.interp(params_to_eval)

            assert np.allclose(result, 1)

        def it_interpolates_a_weird_func_correctly():
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=4, lb=-np.ones(2), ub=np.ones(2))

            def func(x): return x[:, 0] * x[:, 1] * np.sin(x[:, 0])
            func_vals = func(smolyak_grid.grid)

            interpolator = SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            p = np.linspace(0, 1, 20)
            params_to_eval = np.stack((p, p)).T

            result = interpolator.interp(params_to_eval)
            true_vals = func(params_to_eval)

            assert np.allclose(result, true_vals)
