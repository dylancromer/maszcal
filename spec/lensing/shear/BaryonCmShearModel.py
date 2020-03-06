from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
from maszcal.lensing.shear import BaryonCmShearModel


class FakeProjector:
    @staticmethod
    def esd(rs, rho_func):
        rhos = rho_func(rs)
        return np.ones(rhos.shape)


@dataclass
class FakeConModel:
    mass_def: str
    cosmology: object = 'blah'

    def c(self, masses, zs, mass_def):
        return 3*np.ones((masses.size, zs.size))


def describe_gaussian_baryonic_model():

    def describe_math():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.lensing.shear.ConModel', new=FakeConModel)
            mocker.patch('maszcal.lensing.shear.projector', new=FakeProjector)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return BaryonCmShearModel(mus, zs)

        def it_can_calculate_a_gnfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            rhos = baryon_model.rho_bary(radii, mus, alphas, betas, gammas)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)

            rhos = baryon_model.rho_cdm(radii, mus)

            assert np.all(rhos > 0)

        def it_has_the_correct_baryon_fraction(baryon_model):
            rs = np.linspace(
                baryon_model.MIN_INTEGRATION_RADIUS,
                baryon_model.MAX_INTEGRATION_RADIUS,
                baryon_model.NUM_INTEGRATION_RADII
            )
            mus = np.log(1e14)*np.ones(1)
            alphas = 0.88*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)

            rho_barys = baryon_model.rho_bary(rs, mus, alphas, betas, gammas)
            rho_cdms = np.moveaxis(baryon_model.rho_cdm(rs, mus)[..., None], 2, 0)

            ratio = np.trapz(
                rho_barys * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            ) / np.trapz(
                (rho_barys + rho_cdms) * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            )

            f_b = baryon_model.baryon_frac
            assert np.allclose(ratio, f_b)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)

            ds = baryon_model.delta_sigma_cdm(radii, mus)

            assert np.all(ds > 0)

        def it_can_calculate_a_gnfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_total_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_total(radii, mus, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_stacked_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)
            a_szs = np.zeros(3)

            baryon_model._init_stacker()
            baryon_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (baryon_model.mus.size, baryon_model.zs.size)
            )

            ds = baryon_model.stacked_delta_sigma(radii, alphas, betas, gammas, a_szs)

            assert np.all(ds > 0)

    def describe_units():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.lensing.shear.ConModel', new=FakeConModel)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return BaryonCmShearModel(mus, zs, units=u.Msun/u.pc**2)

        def it_has_correct_units(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, alphas, betas, gammas)

            assert np.all(radii[None, None, :, None]*ds < 1e2)