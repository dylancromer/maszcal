import numpy as np
import pytest
from maszcal.emulator import LensingEmulator, LargeErrorWarning




class FakeInterpolator:
    def __init__(self, coords, grid):
        pass

    def process(self):
        pass

    def interp(self, coords):
        return np.ones(tuple(c.size for c in coords))


def describe_emulator():

    def describe_error_check():

        @pytest.fixture
        def emulator(mocker):
            mocker.patch('maszcal.emulator.RbfInterpolator', new=FakeInterpolator)
            lensing_emulator = LensingEmulator()
            lensing_emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))
            return lensing_emulator

        def it_does_nothing_when_the_interpolation_is_good(emulator):
            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            coords = (rs, cons, a_szs)
            emulator.emulate(coords)

        def it_complains_when_the_interpolation_is_bad(emulator):
            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            coords = (rs, cons, a_szs)

            emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))
            emulator.emulate(coords, check_errs=False)
            emulator.generate_grid = lambda coords: 2*np.ones(tuple(c.size for c in coords))

            with pytest.raises(LargeErrorWarning):
                emulator.check_errors(coords)
