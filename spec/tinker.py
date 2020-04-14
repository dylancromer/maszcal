from dataclasses import dataclass
import pytest
import numpy as np
from astropy.cosmology import Planck15
import maszcal.tinker


def describe_TinkerBias():

    @pytest.fixture
    def bias_func():
        delta = 500
        mass_definition = 'crit'
        return maszcal.tinker.TinkerBias(delta, mass_definition, astropy_cosmology=Planck15, comoving=True)

    def it_calculates_dn_dlnm(bias_func):
        masses = np.logspace(14, 15, 10)
        zs = np.linspace(0, 2, 8)
        ks = np.logspace(-3, -1, 12)
        power_spect = np.ones((zs.size, ks.size))

        biases = bias_func.bias(masses, zs, ks, power_spect)

        assert not np.any(np.isnan(biases))

    def delta_ms_are_always_bigger_than_delta_cs(bias_func):
        zs = np.linspace(0, 10, 100)

        delta_ms = bias_func._get_delta_means(zs)

        assert np.all(delta_ms > bias_func.delta)
        assert np.all(delta_ms < 3200)


def describe_TinkerHmf():

    @pytest.fixture
    def mass_func():
        delta = 500
        mass_definition = 'crit'
        return maszcal.tinker.TinkerHmf(delta, mass_definition, astropy_cosmology=Planck15, comoving=True)

    def it_calculates_dn_dlnm(mass_func):
        masses = np.logspace(14, 15, 10)
        zs = np.linspace(0, 2, 8)
        ks = np.logspace(-3, -1, 12)
        power_spect = np.ones((zs.size, ks.size))

        dn_dlnms = mass_func.dn_dlnm(masses, zs, ks, power_spect)

        assert not np.any(np.isnan(dn_dlnms))

    def delta_ms_are_always_bigger_than_delta_cs(mass_func):
        zs = np.linspace(0, 10, 100)

        delta_ms = mass_func._get_delta_means(zs)

        assert np.all(delta_ms > mass_func.delta)
        assert np.all(delta_ms < 3200)