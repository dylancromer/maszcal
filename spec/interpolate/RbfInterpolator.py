import pytest
import numpy as np
from pretend import stub
from maszcal.interpolate import RbfInterpolator, SavedRbf




def describe_rbf_interpolator():

    def describe_init():

        def correct_args_case():
            grid = stub()
            coords = stub()
            RbfInterpolator(coords, grid)

        def incorrect_args_case():
            with pytest.raises(TypeError):
                RbfInterpolator()

    def describe_process():

        def it_creates_an_rbfi():
            coords = (np.linspace(0, 1, 5),)
            grid = np.ones(5)

            interpolator = RbfInterpolator(coords, grid)

            interpolator.process()

            assert isinstance(interpolator.rbfi.nodes, np.ndarray)

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            coords = (np.linspace(0, 1, 10),)
            grid = np.ones(10)

            interpolator = RbfInterpolator(coords, grid)
            interpolator.process()

            coords_to_eval = (np.linspace(0.2, 0.3, 10),)
            result = interpolator.interp(coords_to_eval)

            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            coords = tuple(np.linspace(0, 1, 2) for i in range(10))
            grid = np.ones(tuple(2 for i in range(10)))

            interpolator = RbfInterpolator(coords, grid)
            interpolator.process()

    def describe_get_rbf_solution():

        @pytest.fixture
        def rbf():
            coords = (np.linspace(0, 1, 10),)
            grid = np.ones(10)
            interpolator = RbfInterpolator(coords, grid)
            interpolator.process()
            return interpolator

        def it_gets_everything_needed_to_build_the_rbf(rbf):
            rbf_solution = rbf.get_rbf_solution()
            assert isinstance(rbf_solution, SavedRbf)