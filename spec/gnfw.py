import pytest
import numpy as np
import astropy.units as u
import maszcal.gnfw
import maszcal.cosmology
import maszcal.nfw
import maszcal.concentration


def describe_Gnfw():

    @pytest.fixture
    def nfw_model():
        return maszcal.nfw.NfwModel()

    @pytest.fixture
    def gnfw_model(nfw_model):
        return maszcal.gnfw.Gnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            nfw_model=nfw_model,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 6)
        mus = np.linspace(32, 33, 5)
        zs = np.linspace(0, 1, 4)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_bary = gnfw_model.rho_bary(rs, zs, mus, cons, alphas, betas, gammas)
        rho_cdm = gnfw_model.rho_cdm(rs, zs, mus, cons)
        assert np.all(rho_bary > 0)
        assert np.all(rho_cdm > 0)

    def it_has_the_correct_baryon_fraction(gnfw_model):
        rs = np.linspace(
            gnfw_model.MIN_INTEGRATION_RADIUS,
            gnfw_model.MAX_INTEGRATION_RADIUS,
            gnfw_model.NUM_INTEGRATION_RADII
        )
        zs = np.linspace(0, 1, 8)
        mus = np.log(1e14)*np.ones(1)
        cs = 3*np.ones(1)
        alphas = 0.88*np.ones(1)
        betas = 3.8*np.ones(1)
        gammas = 0.2*np.ones(1)

        rho_barys = gnfw_model.rho_bary(rs, zs, mus, cs, alphas, betas, gammas)
        rho_cdms = np.moveaxis(gnfw_model.rho_cdm(rs, zs, mus, cs), 2, 0)

        ratio = np.trapz(
            rho_barys * rs[:, None, None, None]**2,
            x=rs,
            axis=0
        ) / np.trapz(
            (rho_barys + rho_cdms) * rs[:, None, None, None]**2,
            x=rs,
            axis=0
        )

        f_b = gnfw_model.baryon_frac
        assert np.allclose(ratio, f_b)


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


def describe_CmGnfw():

    @pytest.fixture
    def nfw_model():
        return maszcal.nfw.NfwCmModel()

    @pytest.fixture
    def gnfw_model(nfw_model):
        return maszcal.gnfw.CmGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            nfw_model=nfw_model,
            con_class=FakeConModel,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 6)
        mus = np.linspace(32, 33, 5)
        zs = np.linspace(0, 1, 4)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_bary = gnfw_model.rho_bary(rs, zs, mus, alphas, betas, gammas)
        rho_cdm = gnfw_model.rho_cdm(rs, zs, mus)
        assert np.all(rho_bary > 0)
        assert np.all(rho_cdm > 0)

    def it_has_the_correct_baryon_fraction(gnfw_model):
        rs = np.linspace(
            gnfw_model.MIN_INTEGRATION_RADIUS,
            gnfw_model.MAX_INTEGRATION_RADIUS,
            gnfw_model.NUM_INTEGRATION_RADII
        )
        zs = np.linspace(0, 1, 8)
        mus = np.log(1e14)*np.ones(1)
        alphas = 0.88*np.ones(1)
        betas = 3.8*np.ones(1)
        gammas = 0.2*np.ones(1)

        rho_barys = gnfw_model.rho_bary(rs, zs, mus, alphas, betas, gammas)
        rho_cdms = np.moveaxis(gnfw_model.rho_cdm(rs, zs, mus)[..., None], 2, 0)

        ratio = np.trapz(
            rho_barys * rs[:, None, None, None]**2,
            x=rs,
            axis=0
        ) / np.trapz(
            (rho_barys + rho_cdms) * rs[:, None, None, None]**2,
            x=rs,
            axis=0
        )

        f_b = gnfw_model.baryon_frac
        assert np.allclose(ratio, f_b)
