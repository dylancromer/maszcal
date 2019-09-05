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
        cosmo = CosmoParams(omega_matter=0.4)
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
        nfw_model_500c = NfwModel(delta=500, mass_def='crit')

        rs = np.logspace(-1, 1, 10)
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.delta_sigma(rs, zs, masses, cons)
        ds_500c = nfw_model_500c.delta_sigma(rs, zs, masses, cons)

        assert np.all(ds < ds_500c)

    def the_mass_definition_must_be_correct():
        with pytest.raises(ValueError):
            NfwModel(mass_def='oweijf')
