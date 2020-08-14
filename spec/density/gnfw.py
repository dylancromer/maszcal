import pytest
import numpy as np
import maszcal.density
import maszcal.cosmology
import maszcal.concentration


def describe_MatchingGnfw():
    @pytest.fixture
    def nfw_class():
        return maszcal.density.MatchingNfwModel

    @pytest.fixture
    def gnfw_model(nfw_class):
        return maszcal.density.MatchingGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=nfw_class,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 5)
        mus = np.linspace(32, 33, 4)
        zs = np.linspace(0, 1, 4)
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

        rho_barys = gnfw_model.rho_bary(rs, zs, mus, cs, alphas, betas, gammas)
        rho_cdms = np.moveaxis(gnfw_model.rho_cdm(rs, zs, mus, cs), 1, 0)

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


class FakeMatchingConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((redshifts.size))


def describe_MatchingCmGnfw():
    @pytest.fixture
    def nfw_class():
        return maszcal.density.MatchingCmNfwModel

    @pytest.fixture
    def gnfw_model(nfw_class):
        return maszcal.density.MatchingCmGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=nfw_class,
            con_class=FakeMatchingConModel,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 5)
        mus = np.linspace(32, 33, 4)
        zs = np.linspace(0, 1, 4)
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

        rho_barys = gnfw_model.rho_bary(rs, zs, mus, alphas, betas, gammas)
        rho_cdms = np.moveaxis(gnfw_model.rho_cdm(rs, zs, mus), 1, 0)

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


def describe_Gnfw():

    @pytest.fixture
    def nfw_class():
        return maszcal.density.NfwModel

    @pytest.fixture
    def gnfw_model(nfw_class):
        return maszcal.density.Gnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=nfw_class,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 6)
        mus = np.linspace(32, 33, 5)
        zs = np.linspace(0, 1, 4)
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
    def nfw_class():
        return maszcal.density.NfwCmModel

    @pytest.fixture
    def gnfw_model(nfw_class):
        return maszcal.density.CmGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=nfw_class,
            con_class=FakeConModel,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 6)
        mus = np.linspace(32, 33, 5)
        zs = np.linspace(0, 1, 4)
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


def describe_SingleMassGnfw():
    @pytest.fixture
    def nfw_class():
        return maszcal.density.SingleMassNfwModel

    @pytest.fixture
    def gnfw_model(nfw_class):
        return maszcal.density.SingleMassGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=nfw_class,
        )

    def it_calculates_the_cdm_and_baryonic_densities(gnfw_model):
        rs = np.logspace(-1, 1, 6)
        mus = np.linspace(32, 33, 3)
        zs = np.linspace(0, 1, 4)
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

        rho_barys = gnfw_model.rho_bary(rs, zs, mus, cons, alphas, betas, gammas)
        rho_cdms = np.moveaxis(gnfw_model.rho_cdm(rs, zs, mus, cons), 1, 0)

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
