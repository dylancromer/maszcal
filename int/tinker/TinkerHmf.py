import pytest
import numpy as np
import camb
from astropy.cosmology import Planck15
import maszcal.matter
from maszcal.tinker import TinkerHmf
import maszcal.cosmology


def describe_tinker_hmf():

    def describe_mcrit_case():

        @pytest.fixture
        def ks_and_power_spect():
            cosmo_params = maszcal.cosmology.CosmoParams()
            power = maszcal.matter.Power(cosmo_params=cosmo_params)
            zs = np.linspace(0, 1, 8)
            ks = np.logspace(-4, np.log10(0.3), 200)
            lin_spect = power.spectrum(ks, zs, is_nonlinear=False)
            return ks, lin_spect

        @pytest.fixture
        def mass_func():
            delta = 500
            mass_definition = 'crit'
            return TinkerHmf(delta, mass_definition, astropy_cosmology=Planck15, comoving=False)

        def it_calculates_dn_dlnm(mass_func, ks_and_power_spect):
            ks, power_spect = ks_and_power_spect
            masses = np.array([1e15, 1e16])
            zs = np.linspace(0, 2, 8)
            dn_dlnms = mass_func.dn_dlnm(masses, zs, ks, power_spect)

            assert dn_dlnms.shape == (2, 8)
            assert np.all(dn_dlnms[0, :] > dn_dlnms[1, :])
