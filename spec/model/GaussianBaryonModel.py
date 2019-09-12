import pytest
import numpy as np
import astropy.units as u
from maszcal.model import GaussianBaryonModel


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 10)
            zs = np.linspace(0, 1, 8)
            return GaussianBaryonModel(mus, zs)

        def it_can_calculate_a_baryonic_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            baryon_vars = 1e-1*np.ones(1)

            ds = baryon_model.delta_sigma_baryon(rs, mus, baryon_vars)

            assert not np.any(np.isnan(ds))
            assert ds.shape == (1, 10, 1)

        def it_can_use_other_units(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mu = np.log(2e14)*np.ones(1)
            baryon_vars = 1e-1*np.ones(1)

            ds = baryon_model.delta_sigma_baryon(rs, mu, baryon_vars)

            baryon_model_other_units = GaussianBaryonModel(
                mu,
                baryon_model.zs,
                units=u.Msun/u.Mpc**2
            )

            ds_other_units = baryon_model_other_units.delta_sigma_baryon(rs, mu, baryon_vars)

            assert np.all(ds < ds_other_units)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            cons = np.linspace(2, 3, 3)

            ds = baryon_model.delta_sigma_nfw(rs, mus, cons)

            assert not np.any(np.isnan(ds))

        def it_can_calculate_a_combined_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 10)
            mus = np.log(2e14)*np.ones(1)
            N_PARAMS = 3
            cons = np.linspace(2, 3, N_PARAMS)
            baryon_vars = 1e-1*np.ones(N_PARAMS)

            ds = baryon_model.delta_sigma_of_mass(rs, mus, cons, baryon_vars)

            assert not np.any(np.isnan(ds))
            assert ds.shape == (1, 8, 10, 3)
