import pytest
import numpy as np
from pretend import stub
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.nothing import NoCoords, NoFuncVals




def describe_rbf_interpolator():

    def describe_init():

        def correct_args_case():
            func_vals = stub()
            params = np.ones((1,1)).T
            RbfInterpolator(params, func_vals)

        def incorrect_args_case():
            with pytest.raises(TypeError):
                RbfInterpolator()

        @pytest.fixture
        def saved_rbf():
            return SavedRbf(dimension=1,
                            norm='euclidean',
                            function='multiquadric',
                            data=np.ones(10),
                            coords=np.linspace(0, 1, 10),
                            epsilon=1,
                            smoothness=0,
                            nodes=np.ones(10))

        def it_can_accept_a_saved_interpolation(saved_rbf):
            interpolator = RbfInterpolator(NoCoords(), NoFuncVals(), saved_rbf=saved_rbf)
            assert interpolator.rbfi is not None

    def describe_process():

        def it_creates_an_rbfi():
            p = np.linspace(1, 3, 5)
            params = np.stack((p,p)).T

            func_vals = np.ones(5)

            interpolator = RbfInterpolator(params, func_vals)

            interpolator.process()

            assert isinstance(interpolator.rbfi.nodes, np.ndarray)

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)[:, None]

            func_vals = np.ones((10, 1))

            interpolator = RbfInterpolator(params, func_vals)
            interpolator.process()

            params_to_eval = np.linspace(0, 1, 20)[:, None]

            result = interpolator.interp(params_to_eval)

            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            p = np.arange(1, 5)
            params = np.stack((p, p, p, p, p, p)).T
            func_vals = np.ones(4)

            interpolator = RbfInterpolator(params, func_vals)
            interpolator.process()

    def describe_get_rbf_solution():

        @pytest.fixture
        def rbf():
            params = np.linspace(1, 0, 10)[:, None]
            func_vals = np.ones(10)

            interpolator = RbfInterpolator(params, func_vals)
            interpolator.process()
            return interpolator

        def it_gets_everything_needed_to_build_the_rbf(rbf):
            rbf_solution = rbf.get_rbf_solution()
            assert isinstance(rbf_solution, SavedRbf)
