import pytest
import numpy as np
import maszcal.fitutils


def describe_minimize():

    @pytest.fixture
    def polyfunc():
        return lambda x: (x[0] - 1)**2 * (x[1] - 2)**2 * (x[2] - 3)**2

    def it_can_minimize_a_3d_polynomial_with_global_diffy_evo(polyfunc):
        result = maszcal.fitutils.minimize(
            polyfunc,
            param_mins=np.array([0, 0, 0]),
            param_maxes=np.array([4, 4, 4]),
            method='global-differential-evolution',
        )

        assert np.allclose(result, np.array([1, 2, 3]))
