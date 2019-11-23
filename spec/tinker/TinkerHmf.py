from dataclasses import dataclass
import pytest
import numpy as np
from astropy.cosmology import Planck15
from maszcal.tinker import TinkerHmf
import maszcal.nothing


def describe_tinker_hmf():

    @pytest.fixture
    def mass_func():
        delta = 500
        mass_definition = 'crit'
        return TinkerHmf(delta, mass_definition, astropy_cosmology=Planck15, comoving=True)

    def it_calculates_dn_dlnm(mass_func):
        masses = np.logspace(14, 15, 10)
        zs = np.linspace(0, 1, 8)
        ks = np.logspace(-3, -1, 12)
        power_spect = np.ones((zs.size, ks.size))

        dn_dlnms = mass_func.dn_dlnm(masses, zs, ks, power_spect)

        assert not np.any(np.isnan(dn_dlnms))
