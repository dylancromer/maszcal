import pytest
import numpy as np
from pretend import stub
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.nothing import NoCoords, NoGrid




def describe_rbf_interpolator():

    def describe_init():

        def correct_args_case():
            grid = stub()
            rs = np.ones(1)
            params = np.ones((1,1))
            RbfInterpolator(rs, params, grid)

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
            interpolator = RbfInterpolator(NoCoords(), NoCoords(), NoGrid(), saved_rbf=saved_rbf)
            assert interpolator.rbfi is not None

    def describe_process():

        def it_creates_an_rbfi():
            rs = np.linspace(0, 2, 4)
            p = np.linspace(1, 3, 5)
            params = np.stack((p,p)).T

            grid = np.ones((4, 5))

            interpolator = RbfInterpolator(rs, params, grid)

            interpolator.process()

            assert isinstance(interpolator.rbfi.nodes, np.ndarray)

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            rs = np.linspace(0, 1, 10)
            params = np.ones((1, 2))

            grid = np.ones((10, 1))

            interpolator = RbfInterpolator(rs, params, grid)
            interpolator.process()

            rs_to_eval = np.linspace(0.2, 0.3, 10)
            result = interpolator.interp(rs_to_eval, params)

            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            rs = np.linspace(0, 1, 5)
            p = np.arange(1, 5)
            params = np.stack((p, p, p, p, p, p)).T
            grid = np.ones((5, 4))

            interpolator = RbfInterpolator(rs, params, grid)
            interpolator.process()

    def describe_get_rbf_solution():

        @pytest.fixture
        def rbf():
            rs = np.linspace(0, 1, 10)
            params = np.ones((1, 2))

            grid = np.ones((10, 1))
            interpolator = RbfInterpolator(rs, params, grid)
            interpolator.process()
            return interpolator

        def it_gets_everything_needed_to_build_the_rbf(rbf):
            rbf_solution = rbf.get_rbf_solution()
            assert isinstance(rbf_solution, SavedRbf)
