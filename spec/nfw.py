
import numpy as np
import pytest
import astropy.units as u
from maszcal.cosmology import CosmoParams
import maszcal.nfw


def describe_NfwCmModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.nfw.NfwCmModel(cosmo_params=cosmo)

    def it_calculates_delta_sigma(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds > 0)

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
        return maszcal.nfw.NfwCmModel(cosmo_params=cosmo)

    def it_can_use_different_cosmologies(nfw_model, nfw_model_alt_cosmo):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_alt_cosmo = nfw_model_alt_cosmo.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds != ds_alt_cosmo)

    def it_can_use_different_units(nfw_model):
        nfw_model_other_units = maszcal.nfw.NfwCmModel(units=u.Msun/u.Mpc**2)

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_other_units = nfw_model_other_units.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds_other_units > ds)

    def it_can_use_different_mass_definitions(nfw_model):
        nfw_model_500c = maszcal.nfw.NfwCmModel(delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def it_must_use_a_correct_mass_definition():
        with pytest.raises(ValueError):
            maszcal.nfw.NfwCmModel(mass_definition='oweijf')

    def it_can_use_comoving_coordinates(nfw_model):
        nfw_model_nocomoving = maszcal.nfw.NfwCmModel(units=u.Msun/u.pc**2, comoving=False)

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(1, 2, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        ds_comoving = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_nocomoving = nfw_model_nocomoving.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds_comoving < ds_nocomoving)

    def it_works_for_redshift_0(nfw_model):
        rs = np.logspace(-1, 1, 10)
        z = np.zeros(1)
        m = 2e14*np.ones(1)
        c = 3*np.ones(1)

        ds = nfw_model.delta_sigma(rs, z, m, c)

        assert not np.any(np.isnan(ds))

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 5)
        cons = np.stack((cons, cons, cons)).T

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)


def describe_NfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.nfw.NfwModel(cosmo_params=cosmo)

    def it_calculates_delta_sigma(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (5, 3, 10, 6)

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
        return maszcal.nfw.NfwModel(cosmo_params=cosmo)

    def it_can_use_different_cosmologies(nfw_model, nfw_model_alt_cosmo):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_alt_cosmo = nfw_model_alt_cosmo.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds != ds_alt_cosmo)

    def it_can_use_different_units(nfw_model):
        nfw_model_other_units = maszcal.nfw.NfwModel(units=u.Msun/u.Mpc**2)

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_other_units = nfw_model_other_units.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds_other_units > ds)

    def it_can_use_different_mass_definitions(nfw_model):
        nfw_model_500c = maszcal.nfw.NfwModel(delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def it_must_use_a_correct_mass_definition():
        with pytest.raises(ValueError):
            maszcal.nfw.NfwModel(mass_definition='oweijf')

    def it_can_use_comoving_coordinates(nfw_model):
        nfw_model_nocomoving = maszcal.nfw.NfwModel(units=u.Msun/u.pc**2, comoving=False)

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(1, 2, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds_comoving = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_nocomoving = nfw_model_nocomoving.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds_comoving < ds_nocomoving)

    def it_works_for_redshift_0(nfw_model):
        rs = np.logspace(-1, 1, 10)
        z = np.zeros(1)
        m = 2e14*np.ones(1)
        c = 3*np.ones(1)

        ds = nfw_model.delta_sigma(rs, z, m, c)

        assert not np.any(np.isnan(ds))

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == masses.shape + zs.shape + rs.shape + cons.shape


def describe_SingleMassNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.nfw.SingleMassNfwModel(cosmo_params=cosmo)

    def it_calculates_delta_sigma(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 6)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == zs.shape + rs.shape + cons.shape

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 6)
        cons = np.linspace(2, 4, 6)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == zs.shape + rs.shape + cons.shape
