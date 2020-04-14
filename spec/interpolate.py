import pytest
from pretend import stub
import numpy as np
import smolyak
import maszcal.interpolate


def describe_Rbf():

    @pytest.fixture
    def saved_rbf():
        return maszcal.interpolate.SavedRbf(dimension=1,
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
        rbf = maszcal.interpolate.Rbf(saved_rbf=saved_rbf)
        true_nodes = np.array([0.14783506, -0.07641081,  0.02277437, -0.00811247,  0.00127305,
                               0.00127305, -0.00811247,  0.02277437, -0.07641081,  0.14783506])

        assert np.all(rbf.nodes == true_nodes)

    def it_can_interpolate_a_constant_with_a_saved_rbf(saved_rbf):
        rbf = maszcal.interpolate.Rbf(saved_rbf=saved_rbf)

        coords = np.linspace(0.2, 0.3, 10)

        assert np.allclose(rbf(coords), np.ones(10), rtol=1e-2)


def describe_RbfInterpolator():

    def describe_from_saved_rbf():

        @pytest.fixture
        def saved_rbf():
            return maszcal.interpolate.SavedRbf(dimension=1,
                            norm='euclidean',
                            function='multiquadric',
                            data=np.ones(10),
                            coords=np.linspace(0, 1, 10),
                            epsilon=1,
                            smoothness=0,
                            nodes=np.ones(10))

        def it_works(saved_rbf):
            interpolator = maszcal.interpolate.RbfInterpolator.from_saved_rbf(saved_rbf)
            assert isinstance(interpolator, maszcal.interpolate.RbfInterpolator)

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            params = np.linspace(0, 1, 10)[:, None]

            func_vals = np.ones((10, 1))

            interpolator = maszcal.interpolate.RbfInterpolator(params, func_vals)
            interpolator.process()

            params_to_eval = np.linspace(0, 1, 20)[:, None]

            result = interpolator.interp(params_to_eval)

            assert np.allclose(result, 1, rtol=1e-2)

        def it_can_handle_lots_of_coords():
            p = np.arange(1, 5)
            params = np.stack((p, p, p, p, p, p)).T
            func_vals = np.ones(4)

            interpolator = maszcal.interpolate.RbfInterpolator(params, func_vals)
            interpolator.process()


def describe_SavedRbf():

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
        return maszcal.interpolate.SavedRbf(
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

        with pytest.raises(maszcal.interpolate.RbfMismatchError):
            saved_rbf =  maszcal.interpolate.SavedRbf(
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

        with pytest.raises(maszcal.interpolate.RbfMismatchError):
            saved_rbf =  maszcal.interpolate.SavedRbf(
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
            saved_rbf =  maszcal.interpolate.SavedRbf(
                dimension,
                norm,
                function,
                data,
                coords,
                epsilon,
                smoothness,
                nodes,
            )


def describe_SmolyakInterpolator():

    def describe_init():

        def correct_args_case():
            func_vals = stub()
            smolyak_grid = stub()
            maszcal.interpolate.SmolyakInterpolator(smolyak_grid, func_vals)

        def incorrect_args_case():
            with pytest.raises(TypeError):
                maszcal.interpolate.SmolyakInterpolator()

    def describe_interp():

        def it_interpolates_a_constant_correctly():
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=3, lb=np.zeros(2), ub=np.ones(2))

            func_vals = np.ones((smolyak_grid.grid.shape[0], 2))

            interpolator = maszcal.interpolate.SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            p = np.linspace(0, 1, 20)
            params_to_eval = np.stack((p, p)).T

            result = interpolator.interp(params_to_eval)

            assert np.allclose(result, 1)

        def it_interpolates_a_weird_func_correctly():
            smolyak_grid = smolyak.grid.SmolyakGrid(d=2, mu=4, lb=-np.ones(2), ub=np.ones(2))

            def func(x): return x[:, 0] * x[:, 1] * np.sin(x[:, 0])
            func_vals = func(smolyak_grid.grid)

            interpolator = maszcal.interpolate.SmolyakInterpolator(smolyak_grid, func_vals)
            interpolator.process()

            p = np.linspace(0, 1, 20)
            params_to_eval = np.stack((p, p)).T

            result = interpolator.interp(params_to_eval)
            true_vals = func(params_to_eval)

            assert np.allclose(result, true_vals)
