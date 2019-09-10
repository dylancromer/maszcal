import pytest
import numpy as np
from maszcal.model import GaussianBaryonicModel


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model():
            return GaussianBaryonicModel()

        def it_can_calculate_baryonic_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            baryon_vars = 1e-1*np.ones(1)

            ds = baryon_model.delta_sigma_of_mass(rs, mus, baryon_vars)

            assert not np.any(np.isnan(ds))
