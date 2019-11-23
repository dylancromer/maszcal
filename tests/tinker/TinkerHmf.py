import pytest
import numpy as np
import camb
from astropy.cosmology import Planck15
from maszcal.tinker import TinkerHmf
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_camb_params


def describe_tinker_hmf():

    @pytest.fixture
    def ks_and_power_spect():
        zs = np.linspace(0, 1, 8)
        params = get_camb_params(CosmoParams(), 0.15, zs)

        results = camb.get_results(params)

        ks, _, power_spect = results.get_matter_power_spectrum(minkh=1e-5,
                                                                 maxkh=0.15,
                                                                 npoints=200)
        return ks, power_spect

    @pytest.fixture
    def mass_func(ks_and_power_spect):
        ks, power_spect = ks_and_power_spect
        delta = 500
        mass_definition = 'crit'
        return TinkerHmf(delta, mass_definition, astropy_cosmology=Planck15, ks=ks, power_spect=power_spect)

    def it_calculates_dn_dlnm(mass_func):
        masses = np.array([1e15, 1e16])
        zs = np.linspace(0, 1, 8)
        dn_dlnms = mass_func.dn_dlnm(masses, zs)

        assert np.all(dn_dlnms[0, :] > dn_dlnms[1, :])
