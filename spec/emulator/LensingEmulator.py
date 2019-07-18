import os
import numpy as np
import pytest
from maszcal.emulator import LensingEmulator
from maszcal.interpolate import SavedRbf
from maszcal.interp_utils import cartesian_prod




class FakeInterpolator:
    def __init__(self, params, func_vals):
        pass

    def process(self):
        pass

    def interp(self, params):
        return np.ones(params.shape[0])


def describe_emulator():

    def describe_load_emulation():

        @pytest.fixture
        def emulator():
            lensing_emulator = LensingEmulator()
            lensing_emulator.generate_func_vals = lambda params: np.ones(params.shape[0])
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
            emulator.load_emulation(saved_rbf=saved_rbf)
            assert emulator.interpolator is not None
            assert emulator.interpolator.rbfi is not None

        def it_fails_if_you_dont_give_it_a_file_or_saved_rbf(emulator):
            with pytest.raises(TypeError):
                emulator.load_emulation()

    def describe_save_emulation():

        @pytest.fixture
        def emulator():
            lensing_emulator = LensingEmulator()
            lensing_emulator.generate_func_vals = lambda params: np.ones(params.shape[0])
            return lensing_emulator

        def it_creates_a_saved_file_with_the_interpolation(emulator):
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            params = cartesian_prod(cons, a_szs)
            func_vals = np.ones(25)

            emulator.emulate(params, func_vals)

            rbf = emulator.save_emulation()

            SAVE_FILE = 'data/test/saved_rbf_test.json'

            emulator.save_emulation(SAVE_FILE)

            saved_rbf = emulator._load_interpolation(SAVE_FILE)

            for original_val,json_val in zip(rbf.__dict__.values(),saved_rbf.__dict__.values()):
                if isinstance(json_val, np.ndarray):
                    assert np.all(json_val == original_val)
                else:
                    assert original_val == json_val

            os.remove(SAVE_FILE)
