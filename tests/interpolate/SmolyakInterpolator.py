
import pytest
import numpy as np
import smolyak
from pretend import stub
from maszcal.interpolate import SmolyakInterpolator
from maszcal.interp_utils import cartesian_prod
import maszcal.lensing as lensing


def describe_smolyak_interpolator():

    def describe_interp():

        @pytest.fixture
        def nfw_lensing_signal():
            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return lensing.StackedLensingSignal(log_masses=mus, redshifts=zs)

        @pytest.fixture
        def bary_lensing_signal():
            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return lensing.StackedBaryonLensingSignal(log_masses=mus, redshifts=zs)

        def it_interpolates_the_nfw_stack_correctly(nfw_lensing_signal):
            param_mins = np.array([1, -2])
            param_maxes = np.array([8, 2])
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=6, lb=param_mins, ub=param_maxes)
            rs = np.array([1e-1])

            func_vals = nfw_lensing_signal.stacked_esd(rs, smolyak_grid.grid)[0]

            interpolator = SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            cons = np.linspace(2, 7, 20)
            a_szs = np.linspace(-1, 1, 20)
            params_to_eval = cartesian_prod(cons, a_szs)

            result = interpolator.interp(params_to_eval)
            true_vals = nfw_lensing_signal.stacked_esd(rs, params_to_eval)

            assert np.allclose(result, true_vals)

        def it_interpolates_the_baryonic_stack_correctly(bary_lensing_signal):
            param_mins = np.array([2, 0.6, 3.4, 0.1, -2])
            param_maxes = np.array([6, 1, 4.2, 0.3, 2])
            smolyak_grid = smolyak.grid.SmolyakGrid(d=5, mu=6, lb=param_mins, ub=param_maxes)
            rs = np.array([1e-1])

            func_vals = bary_lensing_signal.stacked_esd(rs, smolyak_grid.grid)[0]

            interpolator = SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            cons = np.linspace(3, 5, 4)
            alphas = np.linspace(0.7, 0.9, 4)
            betas = np.linspace(3.6, 4, 4)
            gammas = np.linspace(0.15, 0.25, 4)
            a_szs = np.linspace(-1, 1, 4)
            params_to_eval = cartesian_prod(cons, alphas, betas, gammas, a_szs)

            result = interpolator.interp(params_to_eval)
            true_vals = bary_lensing_signal.stacked_esd(rs, params_to_eval)

            assert np.allclose(result, true_vals)
