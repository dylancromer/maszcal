import pytest
import numpy as np
from pretend import stub
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.nothing import NoCoords, NoFuncVals


def describe_rbf_interpolator():

    def describe_from_saved_rbf():

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

        def it_works(saved_rbf):
            interpolator = RbfInterpolator.from_saved_rbf(saved_rbf)
            assert isinstance(interpolator, RbfInterpolator)

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
