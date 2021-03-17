import pytest
import numpy as np
import astropy.units as u
import maszcal.density
import maszcal.cosmology
import maszcal.concentration


def describe_MatchingGnfw():
    @pytest.fixture
    def gnfw_model():
        return maszcal.density.MatchingGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            units=u.Msun/u.pc**2,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_tot = gnfw_model.rho_tot(rs, zs, mus, cons, alphas, betas, gammas)
        assert rho_tot.shape == (5, 4, 3)
        assert np.all(rho_tot > 0)

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

        rho_barys = gnfw_model.rho_bary(rs[:, None], zs, mus, cs, alphas, betas, gammas)
        rho_cdms = gnfw_model.rho_cdm(rs[:, None], zs, mus, cs)

        ratio = np.trapz(
            rho_barys * rs[:, None, None]**2,
            x=rs,
            axis=0
        ) / np.trapz(
            (rho_barys + rho_cdms) * rs[:, None, None]**2,
            x=rs,
            axis=0
        )

        f_b = gnfw_model.baryon_frac
        assert np.allclose(ratio, f_b)

    def it_can_calculate_esd(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        esd = gnfw_model.excess_surface_density(rs, zs, mus, cons, alphas, betas, gammas)
        assert esd.shape == (5, 4, 3)

    def it_can_calculate_convergence(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        conv = gnfw_model.convergence(rs, zs, mus, cons, alphas, betas, gammas)
        assert conv.shape == (5, 4, 3)


class FakeMatchingConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((redshifts.size))


def describe_MatchingCmGnfw():
    @pytest.fixture
    def gnfw_model():
        return maszcal.density.MatchingCmGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            con_class=FakeMatchingConModel,
            units=u.Msun/u.pc**2,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_tot = gnfw_model.rho_tot(rs, zs, mus, alphas, betas, gammas)
        assert rho_tot.shape == (5, 4, 3)
        assert np.all(rho_tot > 0)

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

        rho_barys = gnfw_model.rho_bary(rs[:, None], zs, mus, alphas, betas, gammas)
        rho_cdms = gnfw_model.rho_cdm(rs[:, None], zs, mus)

        ratio = np.trapz(
            rho_barys * rs[:, None, None]**2,
            x=rs,
            axis=0
        ) / np.trapz(
            (rho_barys + rho_cdms[..., None]) * rs[:, None, None]**2,
            x=rs,
            axis=0
        )

        f_b = gnfw_model.baryon_frac
        assert np.allclose(ratio, f_b)

    def it_can_calculate_esd(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        esd = gnfw_model.excess_surface_density(rs, zs, mus, alphas, betas, gammas)
        assert esd.shape == (5, 4, 3)

    def it_can_calculate_convergence(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 4)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        conv = gnfw_model.convergence(rs, zs, mus, alphas, betas, gammas)
        assert conv.shape == (5, 4, 3)


def describe_Gnfw():

    @pytest.fixture
    def gnfw_model():
        return maszcal.density.Gnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            units=u.Msun/u.pc**2,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 6)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 5)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_tot = gnfw_model.rho_tot(rs, zs, mus, cons, alphas, betas, gammas)
        assert rho_tot.shape == (6, 5, 4, 3)
        assert np.all(rho_tot > 0)

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

        rho_barys = gnfw_model.rho_bary(rs[:, None], zs, mus, cs, alphas, betas, gammas)
        rho_cdms = gnfw_model.rho_cdm(rs[:, None], zs, mus, cs)

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

    def it_can_calculate_esd(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 2)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        esd = gnfw_model.excess_surface_density(rs, zs, mus, cons, alphas, betas, gammas)
        assert esd.shape == (5, 2, 4, 3)

    def it_can_calculate_convergence(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 2)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        conv = gnfw_model.convergence(rs, zs, mus, cons, alphas, betas, gammas)
        assert conv.shape == (5, 2, 4, 3)


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


def describe_CmGnfw():

    @pytest.fixture
    def gnfw_model():
        return maszcal.density.CmGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            con_class=FakeConModel,
            units=u.Msun/u.pc**2,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 6)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 5)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_tot = gnfw_model.rho_tot(rs, zs, mus, alphas, betas, gammas)
        assert rho_tot.shape == (6, 5, 4, 3)
        assert np.all(rho_tot > 0)

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

        rho_barys = gnfw_model.rho_bary(rs[:, None], zs, mus, alphas, betas, gammas)
        rho_cdms = gnfw_model.rho_cdm(rs[:, None], zs, mus)[..., None]

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

    def it_can_calculate_esd(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 6)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        esd = gnfw_model.excess_surface_density(rs, zs, mus, alphas, betas, gammas)
        assert esd.shape == (5, 6, 4, 3)

    def it_can_calculate_convergence(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 6)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        conv = gnfw_model.convergence(rs, zs, mus, alphas, betas, gammas)
        assert conv.shape == (5, 6, 4, 3)


def describe_SingleMassGnfw():
    @pytest.fixture
    def gnfw_model():
        return maszcal.density.SingleMassGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            units=u.Msun/u.pc**2,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 6)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 3)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        rho_tot = gnfw_model.rho_tot(rs, zs, mus, cons, alphas, betas, gammas)
        assert rho_tot.shape == (6, 4, 3)
        assert np.all(rho_tot > 0)

    def it_has_the_correct_baryon_fraction(gnfw_model):
        rs = np.linspace(
            gnfw_model.MIN_INTEGRATION_RADIUS,
            gnfw_model.MAX_INTEGRATION_RADIUS,
            gnfw_model.NUM_INTEGRATION_RADII
        )
        zs = np.linspace(0, 1, 8)
        mus = np.log(1e14)*np.ones(1)
        cons = 3*np.ones(1)
        alphas = 0.88*np.ones(1)
        betas = 3.8*np.ones(1)
        gammas = 0.2*np.ones(1)

        rho_barys = gnfw_model.rho_bary(rs[:, None], zs, mus, cons, alphas, betas, gammas)
        rho_cdms = gnfw_model.rho_cdm(rs[:, None], zs, mus, cons)

        ratio = np.trapz(
            rho_barys * rs[:, None, None]**2,
            x=rs,
            axis=0
        ) / np.trapz(
            (rho_barys + rho_cdms) * rs[:, None, None]**2,
            x=rs,
            axis=0
        )

        f_b = gnfw_model.baryon_frac
        assert np.allclose(ratio, f_b)

    def it_can_calculate_esd(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 3)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        esd = gnfw_model.excess_surface_density(rs, zs, mus, cons, alphas, betas, gammas)
        assert esd.shape == (5, 4, 3)

    def it_can_calculate_convergence(gnfw_model):
        zs = np.linspace(0.1, 1, 4)
        rs = np.logspace(-1, 1, 5)[:, None] * np.ones_like(zs)[None, :]
        mus = np.linspace(32, 33, 3)
        cons = np.linspace(2, 3, 3)
        alphas = np.linspace(0.5, 1, 3)
        betas = np.linspace(3, 4, 3)
        gammas = np.linspace(0.1, 0.3, 3)
        conv = gnfw_model.convergence(rs, zs, mus, cons, alphas, betas, gammas)
        assert conv.shape == (5, 4, 3)
