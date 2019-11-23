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
        params = get_camb_params(CosmoParams(), max_k=0.3, zs=zs)

        results = camb.get_results(params)

        ks, z, power_spect = results.get_matter_power_spectrum(minkh=1e-5,
                                                               maxkh=0.3,
                                                               npoints=200)

        return ks, power_spect

    @pytest.fixture
    def mass_func():
        delta = 500
        mass_definition = 'crit'
        return TinkerHmf(delta, mass_definition, astropy_cosmology=Planck15, comoving=False)

    def it_calculates_dn_dlnm(mass_func, ks_and_power_spect):
        ks, power_spect = ks_and_power_spect
        masses = np.array([1e15, 1e16])
        zs = np.linspace(0, 1, 8)
        dn_dlnms = mass_func.dn_dlnm(masses, zs, ks, power_spect)

        assert dn_dlnms.shape == (2, 8)
        assert np.all(dn_dlnms[0, :] > dn_dlnms[1, :])
