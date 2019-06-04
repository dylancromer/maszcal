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


def test_emulator_error_check_passing(mocker):
    emulator = LensingEmulator()

    rs = np.logspace(-1, 1, 10)
    cons = np.linspace(2, 5, 5)
    a_szs = np.linspace(-1, 1, 5)
    coords = (rs, cons, a_szs)

    mocker.patch('maszcal.emulator.RbfInterpolator', new=FakeInterpolator)

    emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))

    emulator.emulate(coords)


def test_emulator_error_check_failing(mocker):
    emulator = LensingEmulator()

    rs = np.logspace(-1, 1, 10)
    cons = np.linspace(2, 5, 5)
    a_szs = np.linspace(-1, 1, 5)
    coords = (rs, cons, a_szs)

    mocker.patch('maszcal.emulator.RbfInterpolator', new=FakeInterpolator)

    emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))

    emulator.emulate(coords, check_errs=False)

    emulator.generate_grid = lambda coords: 2*np.ones(tuple(c.size for c in coords))

    with pytest.raises(LargeErrorWarning):
        emulator.check_errors(coords)
