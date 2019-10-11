import pytest
import numpy as np
from maszcal.lensing import StackedLensingSignal


def describe_lensing_signal():

    def describe_stacked_esd():

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
        def lsignal():
            mus = np.linspace(np.log(1e14), np.log(1e16), 25)
            zs = np.linspace(0, 2, 25)
            return StackedLensingSignal(log_masses=mus, redshifts=zs)

        def its_fast_for_20_radial_bins(radii_20, params, lsignal, benchmark):
            benchmark(lsignal.stacked_esd, radii_20, params)

        def its_fast_for_5_radial_bins(radii_5, params, lsignal, benchmark):
            benchmark(lsignal.stacked_esd, radii_5, params)
