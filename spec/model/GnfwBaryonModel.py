import pytest
import numpy as np
import astropy.units as u
from maszcal.model import GnfwBaryonModel


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return GnfwBaryonModel(mus, zs)

        def it_can_calculate_a_gnfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            alphas = np.ones(1)
            betas = 2*np.ones(1)
            gammas = np.ones(1)

            rhos = baryon_model.rho_gnfw(radii, mus, alphas, betas, gammas)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            rhos = baryon_model.rho_nfw(radii, mus, cs)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            ds = baryon_model.delta_sigma_nfw(radii, mus, cs)

            assert np.all(ds > 0)

        def it_can_calculate_a_gnfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            alphas = np.ones(1)
            betas = 2*np.ones(1)
            gammas = np.ones(1)

            ds = baryon_model.delta_sigma_gnfw(radii, mus, alphas, betas, gammas)

            assert np.all(ds > 0)
