import os
import numpy as np
import pytest
from maszcal.emulator import LensingEmulator, LargeErrorWarning
from maszcal.interpolate import SavedRbf




class FakeInterpolator:
    def __init__(self, coords, grid):
        pass

    def process(self):
        pass

    def interp(self, coords):
        return np.ones(tuple(c.size for c in coords))


def describe_emulator():

    def describe_load_emulation():

        @pytest.fixture
        def emulator():
            lensing_emulator = LensingEmulator()
            lensing_emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))
            return lensing_emulator

        @pytest.fixture
        def saved_rbf():
            return SavedRbf(dimension=1,
                            norm='euclidean',
                            function='multiquadric',
                            data=np.ones(10),
                            coords=np.linspace(0, 1, 10),
                            epsilon=1,
                            smoothness=0,
                            nodes=np.ones(10))

        def it_can_accept_a_saved_interpolator(emulator, saved_rbf):
            emulator.load_emulation(saved_rbf)
            assert emulator.interpolator is not None
            assert emulator.interpolator.rbfi is not None

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

    def describe_save_interpolation():

        @pytest.fixture
        def emulator():
            lensing_emulator = LensingEmulator()
            lensing_emulator.generate_grid = lambda coords: np.ones(tuple(c.size for c in coords))
            return lensing_emulator

        def it_creates_a_saved_file_with_the_interpolation(emulator):
            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            coords = (rs, cons, a_szs)
            emulator.emulate(coords)

            rbf = emulator.save_interpolation()

            SAVE_FILE = 'data/test/saved_rbf_test.json'

            emulator.save_interpolation(SAVE_FILE)

            saved_rbf = emulator.load_interpolation(SAVE_FILE, return_rbf=True)

            for original_val,json_val in zip(rbf.__dict__.values(),saved_rbf.__dict__.values()):
                if isinstance(json_val, np.ndarray):
                    assert np.all(json_val == original_val)
                else:
                    assert original_val == json_val

            os.remove(SAVE_FILE)
