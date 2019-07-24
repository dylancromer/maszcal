import pytest
import numpy as np
from maszcal.lensing import LensingSignal




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
    def lsignal():
        return LensingSignal()

    def its_fast_for_20_radial_bins(radii_20, params, lsignal, benchmark):
        benchmark(lsignal.stacked_esd, radii_20, params)

    def its_fast_for_5_radial_bins(radii_5, params, lsignal, benchmark):
        benchmark(lsignal.stacked_esd, radii_5, params)
