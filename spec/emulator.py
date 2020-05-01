import pytest
import numpy as np
import maszcal.emulator
import pality


def describe_LensingPca():

    def describe_standardize():

        @pytest.fixture
        def data():
            return 2*np.random.randn(5, 10) + 1

        def it_normalizes_the_lensing_data(data):
            standardized_data =  maszcal.emulator.LensingPca.standardize(data)
            assert np.allclose(standardized_data.mean(axis=-1), 0)
            assert np.allclose(standardized_data.std(axis=-1), 1)

    def describe_get_pca():

        @pytest.fixture
        def data():
            return np.random.randn(5, 10)

        def it_retrieves_a_pca_from_pality(data):
            pca = maszcal.emulator.LensingPca.get_pca(data)
            assert isinstance(pca, pality.PcData)

    @pytest.fixture
    def data():
        return 2*np.random.randn(5, 10) + 1

    def it_returns_a_pca_of_the_standardized_data(data):
        pca = maszcal.emulator.LensingPca.create(data)
        reconstructed_data = pca.basis_vectors @ pca.weights
        assert np.allclose(reconstructed_data.mean(axis=-1), 0)
        assert np.allclose(reconstructed_data.std(axis=-1), 1)
