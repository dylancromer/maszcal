
import numpy as np
import pytest
import astropy.units as u
from maszcal.cosmology import CosmoParams
import maszcal.density


def describe_MatchingNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.MatchingNfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 4)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (10, 8, 4)

    def it_can_calculate_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 4)

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert np.all(sd > 0)
        assert sd.shape == (10, 8, 4)

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 8)
        cons = np.linspace(2, 4, 4)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == (10, 8, 4)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0.001, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 4)

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons)

        assert np.all(conv >= 0)
        assert conv.shape == (10, 8, 4)


def describe_MatchingCmNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.MatchingCmNfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 8)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (10, 8)

    def it_calculates_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 8)

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert np.all(sd > 0)
        assert sd.shape == (10, 8)

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 8)
        cons = np.linspace(2, 4, 8)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == (10, 8)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0.001, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 8)

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons)

        assert np.all(conv >= 0)
        assert conv.shape == (10, 8)


def describe_CmNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.CmNfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert ds.shape == (10, 5, 3)
        assert np.all(ds > 0)

    def it_calculates_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert sd.shape == (10, 5, 3)
        assert np.all(sd > 0)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0.001, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 5)
        cons = np.linspace(2, 3.4, 5)
        cons = np.stack((cons, cons, cons)).T

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons)

        assert np.all(conv >= 0)
        assert conv.shape == (10, 5, 3)

    @pytest.fixture
    def nfw_model_alt_cosmo():
        cosmo = CosmoParams(
            omega_matter=0.4,
            omega_cdm=0.3,
            omega_cdm_hsqr=0.7**2 * 0.3,
            omega_bary=0.1,
            omega_bary_hsqr=0.7**2 * 0.1,
            omega_lambda=0.6,
            hubble_constant=70,
            h=0.7,
        )
        return maszcal.density.CmNfwModel(cosmo_params=cosmo)

    def it_can_use_different_cosmologies(nfw_model, nfw_model_alt_cosmo):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_alt_cosmo = nfw_model_alt_cosmo.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds != ds_alt_cosmo)

    def it_can_use_different_units(nfw_model):
        nfw_model_other_units = maszcal.density.CmNfwModel(units=u.Msun/u.Mpc**2)

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_other_units = nfw_model_other_units.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds_other_units > ds)

    def it_can_use_different_mass_definitions(nfw_model):
        nfw_model_500c = maszcal.density.CmNfwModel(delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def it_must_use_a_correct_mass_definition():
        with pytest.raises(ValueError):
            maszcal.density.CmNfwModel(mass_definition='oweijf')

    def it_can_use_comoving_coordinates(nfw_model):
        nfw_model_nocomoving = maszcal.density.CmNfwModel(units=u.Msun/u.pc**2, comoving=False)

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(1, 2, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds_comoving = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_nocomoving = nfw_model_nocomoving.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds_comoving < ds_nocomoving)

    def it_works_for_redshift_0(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        z = np.zeros(1)
        m = 2e14*np.ones(1)
        c = 3*np.ones(1)

        ds = nfw_model.excess_surface_density(rs, z, m, c)

        assert not np.any(np.isnan(ds))

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert rhos.shape == (10, 5, 3)
        assert np.all(rhos > 0)


def describe_NfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.NfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (10, 5, 3, 6)

    def it_calculates_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert np.all(sd > 0)
        assert sd.shape == (10, 5, 3, 6)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0.001, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 5)
        cons = np.linspace(2, 3.4, 6)

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons)

        assert np.all(conv >= 0)
        assert conv.shape == (10, 5, 3, 6)

    @pytest.fixture
    def nfw_model_alt_cosmo():
        cosmo = CosmoParams(
            omega_matter=0.4,
            omega_cdm=0.3,
            omega_cdm_hsqr=0.7**2 * 0.3,
            omega_bary=0.1,
            omega_bary_hsqr=0.7**2 * 0.1,
            omega_lambda=0.6,
            hubble_constant=70,
            h=0.7,
        )
        return maszcal.density.NfwModel(cosmo_params=cosmo)

    def it_can_use_different_cosmologies(nfw_model, nfw_model_alt_cosmo):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_alt_cosmo = nfw_model_alt_cosmo.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds != ds_alt_cosmo)

    def it_can_use_different_units(nfw_model):
        nfw_model_other_units = maszcal.density.NfwModel(units=u.Msun/u.Mpc**2)

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_other_units = nfw_model_other_units.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds_other_units > ds)

    def it_can_use_different_mass_definitions(nfw_model):
        nfw_model_500c = maszcal.density.NfwModel(delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def it_must_use_a_correct_mass_definition():
        with pytest.raises(ValueError):
            maszcal.density.NfwModel(mass_definition='oweijf')

    def it_can_use_comoving_coordinates(nfw_model):
        nfw_model_nocomoving = maszcal.density.NfwModel(units=u.Msun/u.pc**2, comoving=False)

        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(1, 2, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds_comoving = nfw_model.excess_surface_density(rs, zs, masses, cons)
        ds_nocomoving = nfw_model_nocomoving.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds_comoving < ds_nocomoving)

    def it_works_for_redshift_0(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        z = np.zeros(1)
        m = 2e14*np.ones(1)
        c = 3*np.ones(1)

        ds = nfw_model.excess_surface_density(rs, z, m, c)

        assert not np.any(np.isnan(ds))

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == (10, 5, 3, 6)


def describe_SingleMassNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.SingleMassNfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 6)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (10, 3, 6)

    def it_calculates_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 6)
        cons = np.linspace(2, 4, 6)

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert np.all(sd > 0)
        assert sd.shape == (10, 3, 6)

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 6)
        cons = np.linspace(2, 4, 6)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == (10, 3, 6)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0.001, 1, 3)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 6)
        cons = np.linspace(2, 3.4, 6)

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons)

        assert np.all(conv >= 0)
        assert conv.shape == (10, 3, 6)
