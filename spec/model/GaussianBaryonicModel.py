import pytest
import numpy as np
from maszcal.model import GaussianBaryonicModel


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 10)
            zs = np.linspace(0, 1, 8)
            return GaussianBaryonicModel(mus, zs)

        def it_can_calculate_a_baryonic_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            baryon_vars = 1e-1*np.ones(1)

            ds = baryon_model.delta_sigma_baryon(rs, mus, baryon_vars)

            assert not np.any(np.isnan(ds))

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            cons = np.linspace(2, 3, 3)

            ds = baryon_model.delta_sigma_nfw(rs, mus, cons)

            assert not np.any(np.isnan(ds))
