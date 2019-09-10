import numpy as np
import pytest
import astropy.units as u
from maszcal.cosmology import CosmoParams
from maszcal.nfw import NfwModel


def describe_nfw_model():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return NfwModel(cosmo_params=cosmo)

    def it_calculates_delta_sigma(nfw_model):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

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
        return NfwModel(cosmo_params=cosmo)

    def it_can_use_different_cosmologies(nfw_model, nfw_model_alt_cosmo):
        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_alt_cosmo = nfw_model_alt_cosmo.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds != ds_alt_cosmo)

    def it_can_use_different_units(nfw_model):
        nfw_model_other_units = NfwModel(units=u.Msun/u.Mpc**2)

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_other_units = nfw_model_other_units.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds_other_units > ds)

    def it_can_use_different_mass_definitions(nfw_model):
        nfw_model_500c = NfwModel(delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def it_must_use_a_correct_mass_definition():
        with pytest.raises(ValueError):
            NfwModel(mass_definition='oweijf')

    def it_can_use_comoving_coordinates(nfw_model):
        nfw_model_nocomoving = NfwModel(units=u.Msun/u.pc**2, comoving=False)

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
