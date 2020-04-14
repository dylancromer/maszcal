import numpy as np
import pytest
import astropy.units as u
from maszcal.cosmology import CosmoParams
from maszcal.nfw import SingleMassNfwModel


def describe_nfw_model():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return SingleMassNfwModel(cosmo_params=cosmo)

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
