import pytest
import numpy as np
from maszcal.interp_utils import cartesian_prod
from maszcal.lensing import LensingSignal
from maszcal.emulator import LensingEmulator


def describe_lensing_signal():

    @pytest.fixture
    def radii_20():
        return np.logspace(-1, 1, 20)

    @pytest.fixture
    def radii_5():
        return np.logspace(-1, 1, 5)

    @pytest.fixture
    def params():
        a = np.random.rand(1)
        c = 2*np.random.rand(1) + 2
        return np.array([c, a]).T

    @pytest.fixture
    def emulator_20():
        redshifts = np.linspace(0, 2, 30)
        log_masses = np.linspace(np.log(1e14), np.log(1e16), 30)

        lsignal = LensingSignal(log_masses=log_masses, redshifts=redshifts)

        cons = np.linspace(2, 5, 10)
        a_szs = np.linspace(-1.2, 1.2, 10)
        #cent_fracs = np.linspace(0, 1, 5)
        #misc_lengths = np.linspace(1e-4, 1e-2, 5)

        params = cartesian_prod(cons, a_szs)

        rs = np.logspace(-1, 1, 20)

        esd = lsignal.stacked_esd(rs, params)

        emulator = LensingEmulator()
        emulator.emulate(rs, params, esd)
        return emulator

    @pytest.fixture
    def emulator_5():
        redshifts = np.linspace(0, 2, 30)
        log_masses = np.linspace(np.log(1e14), np.log(1e16), 30)

        lsignal = LensingSignal(log_masses=log_masses, redshifts=redshifts)

        cons = np.linspace(2, 5, 10)
        a_szs = np.linspace(-1.2, 1.2, 10)
        #cent_fracs = np.linspace(0, 1, 5)
        #misc_lengths = np.linspace(1e-4, 1e-2, 5)

        params = cartesian_prod(cons, a_szs)

        rs = np.logspace(-1, 1, 5)

        esd = lsignal.stacked_esd(rs, params)

        emulator = LensingEmulator()
        emulator.emulate(rs, params, esd)
        return emulator

    def its_fast_for_20_radial_bins(params, emulator_20, benchmark):
        benchmark(emulator_20.evaluate_on, params)

    def its_fast_for_5_radial_bins(params, emulator_5, benchmark):
        benchmark(emulator_5.evaluate_on, params)
