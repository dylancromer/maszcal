import pytest
import numpy as np
from maszcal.interpolate import SavedRbf, RbfMismatchError




def describe_saved_rbf():

    @pytest.fixture
    def saved_rbf():
        dimension = 1
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 10)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(10)
        return SavedRbf(
            dimension,
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

    def it_makes_sure_nodes_are_always_same_size_as_data():
        dimension = 1
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 10)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(11)

        with pytest.raises(RbfMismatchError):
            saved_rbf =  SavedRbf(
                dimension,
                norm,
                function,
                data,
                coords,
                epsilon,
                smoothness,
                nodes,
            )

    def it_makes_sure_dimensions_coords_and_data_match():
        dimension = 3
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 20)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(10)

        with pytest.raises(RbfMismatchError):
            saved_rbf =  SavedRbf(
                dimension,
                norm,
                function,
                data,
                coords,
                epsilon,
                smoothness,
                nodes,
            )

    def it_makes_sure_dimension_is_int():
        dimension = 2.5
        norm = 'euclidean'
        function = 'multiquadric'
        data = np.ones(10)
        coords = np.linspace(0, 1, 25)
        epsilon = 1
        smoothness = 0
        nodes = np.ones(10)

        with pytest.raises(TypeError):
            saved_rbf =  SavedRbf(
                dimension,
                norm,
                function,
                data,
                coords,
                epsilon,
                smoothness,
                nodes,
            )
