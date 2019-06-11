import pytest
import numpy as np
from maszcal.interpolate import SavedRbf, RbfMismatchError




def describe_saved_rbf():

    @pytest.fixture
    def saved_rbf():
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 10)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(10)
        return SavedRbf(
            norm,
            function,
            data,
            coords,
            epsilon,
            smoothness,
            nodes,
        )

    def it_contains_rbf_nodes(saved_rbf):
        assert saved_rbf.nodes is not None

    def nodes_are_always_same_size_as_data():
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 10)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(11)

        with pytest.raises(RbfMismatchError):
            saved_rbf =  SavedRbf(
                norm,
                function,
                data,
                coords,
                epsilon,
                smoothness,
                nodes,
            )

