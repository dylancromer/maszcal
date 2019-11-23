import pytest
import numpy as np
from maszcal.tinker import TinkerHmf
import maszcal.nothing


class FakeAstropyCosmology:
    def critical_density(self, z):
        return np.ones(z.shape)

    def Om(self, z):
        return np.ones(z.shape)


def describe_tinker_hmf():

    @pytest.fixture
    def mass_func():
        delta = 500
        mass_definition = 'crit'
        return TinkerHmf(delta, mass_definition, astropy_cosmology=FakeAstropyCosmology())

    def it_calculates_dn_dlnm(mass_func):
        masses = np.logspace(14, 15, 10)
        zs = np.linspace(0, 1, 8)
        dn_dlnms = mass_func.dn_dlnm(masses, zs)

        assert not np.any(np.isnan(dn_dlnms))
