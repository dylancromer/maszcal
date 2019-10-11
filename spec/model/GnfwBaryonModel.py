import pytest
import numpy as np
import astropy.units as u
from maszcal.model import GnfwBaryonModel


class FakeProjector:
    @staticmethod
    def esd(rs, rho_func):
        rhos = rho_func(rs)
        return np.ones(rhos.shape)


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.model.projector', new=FakeProjector)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return GnfwBaryonModel(mus, zs)

        def it_can_calculate_a_gnfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)
            alphas = np.ones(1)
            betas = 2*np.ones(1)
            gammas = np.ones(1)

            rhos = baryon_model.rho_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            rhos = baryon_model.rho_cdm(radii, mus, cs)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            ds = baryon_model.delta_sigma_cdm(radii, mus, cs)

            assert np.all(ds > 0)

        def it_can_calculate_a_gnfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_total_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_total(radii, mus, cs, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_stacked_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)
            a_szs = np.zeros(1)

            baryon_model._init_stacker()
            baryon_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (baryon_model.mus.size, baryon_model.zs.size)
            )

            ds = baryon_model.stacked_delta_sigma(radii, cs, alphas, betas, gammas, a_szs)

            assert np.all(ds > 0)

    def describe_units():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return GnfwBaryonModel(mus, zs, units=u.Msun/u.pc**2)

        def it_has_correct_units(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(radii[None, None, :, None]*ds < 1e2)
