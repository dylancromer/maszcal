
import pytest
import numpy as np
from maszcal.lensing import SingleMassLensingSignal
from maszcal.interp_utils import cartesian_prod


def describe_lensing_signal():

    def describe_single_mass_esd():

        @pytest.fixture
        def params_1():
            return np.array([[3, np.log(5e14)]])

        @pytest.fixture
        def params_lots():
            mus = np.linspace(np.log(1e14), np.log(1e16), 10)
            cons = np.linspace(2, 10, 10)
            return cartesian_prod(mus, cons)

        @pytest.fixture
        def radii():
            return np.logspace(-1, 1, 10)

        @pytest.fixture
        def lsignal():
            redshift = np.array([0.4])
            return SingleMassLensingSignal(redshift=redshift)

        def its_fast_for_1_mass_and_con(benchmark, lsignal, radii, params_1):
            benchmark(lsignal.esd, radii, params_1)

        def its_fast_for_lots_of_masses_and_cons(benchmark, lsignal, radii, params_lots):
            benchmark(lsignal.esd, radii, params_lots)
