import os
import pytest
import numpy as np
import maszcal.emulate
import maszcal.interpolate
import pality


def describe_LensingPca():

    def describe_standardize():

        @pytest.fixture
        def data():
            return 2*np.random.randn(10, 5) + 1

        def it_normalizes_the_lensing_data(data):
            standardized_data =  maszcal.emulate.LensingPca.standardize(data)
            assert np.allclose(standardized_data.mean(axis=-1), 0)
            assert np.allclose(standardized_data.std(axis=-1), 1)

    def describe_get_pca():

        @pytest.fixture
        def data():
            return np.random.randn(10, 5)

        def it_retrieves_a_pca_from_pality(data):
            pca = maszcal.emulate.LensingPca.get_pca(data)
            assert isinstance(pca, pality.PcData)

    @pytest.fixture
    def data():
        return 2*np.random.randn(10, 5) + 1

    def it_returns_a_pca_of_the_standardized_data(data):
        pca = maszcal.emulate.LensingPca.create(data)
        reconstructed_data = pca.basis_vectors @ pca.weights
        assert np.allclose(reconstructed_data.mean(axis=-1), 0)
        assert np.allclose(reconstructed_data.std(axis=-1), 1)


def describe_PcaEmulator():

    @pytest.fixture
    def coords():
        return np.linspace(0, 1, 5)

    @pytest.fixture
    def data():
        return 2*np.random.randn(10, 5) + 1

    @pytest.fixture
    def pca(data):
        return maszcal.emulate.LensingPca.create(data)

    def it_emulates_a_pca_to_reproduce_data(coords, data, pca):
        emulator = maszcal.emulate.PcaEmulator(
            mean=data.mean(axis=-1),
            std_dev=data.std(axis=-1),
            coords=coords,
            basis_vectors=pca.basis_vectors,
            explained_variance=pca.explained_variance,
            weights=pca.weights,
            interpolator_class=maszcal.interpolate.RbfInterpolator
        )

        new_coords = np.linspace(0.1, 0.9, 20)
        assert emulator(new_coords).shape == (10, 20)

    def it_can_be_created_from_data_directly(coords, data):
        emulator = maszcal.emulate.PcaEmulator.create_from_data(
            coords=coords,
            data=data,
            interpolator_class=maszcal.interpolate.RbfInterpolator,
            interpolator_kwargs={},
        )

        new_coords = np.linspace(0.1, 0.9, 20)
        assert emulator(new_coords).shape == (10, 20)

    def describe_radial_interpolation():

        @pytest.fixture
        def radial_grid():
            return np.geomspace(1e-1, 1e1, 30)

        @pytest.fixture
        def coords():
            return np.random.rand(10, 2)

        @pytest.fixture
        def data(radial_grid, coords):
            a = coords[:, 0][None, :]
            b = coords[:, 1][None, :]
            return a*radial_grid[:, None] + b

        def it_can_interpolate_over_radii(data, coords, radial_grid):
            emulator = maszcal.emulate.PcaEmulator.create_from_data(
                coords=coords,
                data=data,
                interpolator_class=maszcal.interpolate.RbfInterpolator,
                interpolator_kwargs={},
            )
            new_coords = 0.8*np.random.rand(5, 2) - 0.1
            new_radii = np.linspace(0.5, 8, 20)
            assert not np.any(np.isnan(emulator.with_new_radii(radial_grid, new_radii, new_coords)))
            assert emulator.with_new_radii(radial_grid, new_radii, new_coords).shape == (20, 5)

    def describe_saving():

        @pytest.fixture
        def coords():
            return np.linspace(0, 1, 10)

        @pytest.fixture
        def data():
            return np.ones((30, 10)) + 1e-4*np.random.randn(30, 10)

        def it_can_save_and_load_the_interpolation_parameters(coords, data):
            emulator = maszcal.emulate.PcaEmulator.create_from_data(
                coords=coords,
                data=data,
                interpolator_class=maszcal.interpolate.RbfInterpolator,
                interpolator_kwargs={},
            )
            new_coords = np.linspace(0.2, 0.8, 12)
            assert np.allclose(emulator(new_coords), 1, atol=1e-2)

            maszcal.emulate.save_pca_emulator('test.emulator', emulator)

            new_emulator = maszcal.emulate.load_pca_emulator('test.emulator')
            os.remove('test.emulator')
            assert np.allclose(new_emulator(new_coords), 1, atol=1e-2)
