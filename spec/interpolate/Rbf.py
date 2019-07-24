import pytest
import numpy as np
from maszcal.interpolate import Rbf, SavedRbf


def describe_rbf():

    @pytest.fixture
    def saved_rbf():
        return SavedRbf(dimension=1,
                        norm='euclidean',
                        function='multiquadric',
                        data=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                        coords=np.array([[0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
                                          0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ]]),
                        epsilon=0.1,
                        smoothness=0.0,
                        nodes=np.array([0.14783506, -0.07641081,  0.02277437, -0.00811247,  0.00127305,
                                        0.00127305, -0.00811247,  0.02277437, -0.07641081,  0.14783506]))

    def it_can_accept_a_saved_rbf(saved_rbf):
        rbf = Rbf(saved_rbf=saved_rbf)
        true_nodes = np.array([0.14783506, -0.07641081,  0.02277437, -0.00811247,  0.00127305,
                               0.00127305, -0.00811247,  0.02277437, -0.07641081,  0.14783506])

        assert np.all(rbf.nodes == true_nodes)

    def it_can_interpolate_a_constant_with_a_saved_rbf(saved_rbf):
        rbf = Rbf(saved_rbf=saved_rbf)

        coords = np.linspace(0.2, 0.3, 10)

        assert np.allclose(rbf(coords), np.ones(10), rtol=1e-2)
