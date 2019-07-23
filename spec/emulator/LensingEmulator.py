import os
import numpy as np
import pytest
from pretend import stub
import maszcal.emulator as emulator
from maszcal.interpolate import SavedRbf
from maszcal.interp_utils import cartesian_prod




class FakeInterpolator:
    def __init__(self, params, func_vals):
        pass

    def process(self):
        pass

    def interp(self, params):
        return np.ones(params.shape[0])


def describe_emulation():

    @pytest.fixture
    def emulation():
        return emulator.Emulation(radii=np.logspace(-1, 1, 3),
                         saved_rbfs=[stub(), stub(), stub()])

    def it_throws_an_error_if_radii_are_wrong_shape():
        with pytest.raises(TypeError):
            emulator.Emulation(radii=np.ones((2, 2)), saved_rbfs=[stub(), stub(), stub()])

    def it_throws_an_error_if_radii_dont_match_interps():
        with pytest.raises(ValueError):
            emulator.Emulation(radii=np.ones(3), saved_rbfs=[stub()])


    def describe_save_and_load():
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

        @pytest.fixture
        def emulation(saved_rbf):
            return emulator.Emulation(radii=np.array([1]),
                             saved_rbfs=[saved_rbf])

        def it_save_a_file_with_the_emulation(emulation):
            SAVED_FILE = 'data/test/emulation.test.json'

            emulation.save(SAVED_FILE)

            assert os.path.isfile(SAVED_FILE)

            os.remove(SAVED_FILE)

        def it_can_load_a_saved_emulation_file(emulation):
            SAVED_FILE = 'data/test/emulation.test.json'

            emulation.save(SAVED_FILE)

            new_emulation = emulator.Emulation.load(SAVED_FILE)

            assert np.all(new_emulation.radii == emulation.radii)

            os.remove(SAVED_FILE)


def describe_emulator():

    def describe_emulate():

        @pytest.fixture
        def lensing_emulator():
            return emulator.LensingEmulator()

        def it_works_over_many_radii(lensing_emulator):
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            params = cartesian_prod(cons, a_szs)

            rs = np.logspace(-1, 1, 10)

            func_vals = np.ones((10, 25))

            lensing_emulator.emulate(rs, params, func_vals)

            test_params = np.array([[3, 0]])

            test_value = lensing_emulator.evaluate_on(rs, test_params)

            assert np.allclose(test_value, np.ones(10), rtol=1e-2)
            assert test_value.shape == (10, 1)

    def describe_load_emulation():

        @pytest.fixture
        def lensing_emulator():
            lensing_emulator = emulator.LensingEmulator()
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

        @pytest.fixture
        def emulation(saved_rbf):
            return emulator.Emulation(radii=np.linspace(0, 1, 2),
                                      saved_rbfs=[saved_rbf, saved_rbf])

        def it_can_accept_a_saved_emulation(lensing_emulator, emulation):
            lensing_emulator.load_emulation(saved_emulation=emulation)
            assert lensing_emulator.interpolators is not None
            assert lensing_emulator.interpolators[0].rbfi is not None

        def it_fails_if_you_dont_give_it_a_file_or_emulation(lensing_emulator):
            with pytest.raises(TypeError):
                lensing_emulator.load_emulation()

    def describe_save_emulation():

        @pytest.fixture
        def lensing_emulator():
            lensing_emulator = emulator.LensingEmulator()
            return lensing_emulator

        def it_creates_a_saved_file_with_the_interpolation(lensing_emulator):
            cons = np.linspace(2, 5, 5)
            a_szs = np.linspace(-1, 1, 5)
            params = cartesian_prod(cons, a_szs)

            rs = np.ones(1)

            func_vals = np.ones((1, 25))

            lensing_emulator.emulate(rs, params, func_vals)

            emulation = lensing_emulator.save_emulation()

            SAVE_FILE = 'data/test/saved_rbf_test.json'

            lensing_emulator.save_emulation(SAVE_FILE)

            saved_emulation = emulator.Emulation.load(SAVE_FILE)

            def _check_equal(correct, should_be_correct):
                #TODO: make this less horrific
                for original_val,json_val in zip(correct.__dict__.values(), should_be_correct.__dict__.values()):
                    if isinstance(json_val, np.ndarray):
                        assert np.all(json_val == original_val)

                    elif isinstance(json_val, SavedRbf):
                        _check_equal(original_val, json_val)

                    elif isinstance(json_val, list):
                        for orig,json in zip(original_val, json_val):
                            _check_equal(orig, json)

                    else:
                        print(original_val)
                        assert original_val == json_val

            _check_equal(emulation, saved_emulation)

            os.remove(SAVE_FILE)
