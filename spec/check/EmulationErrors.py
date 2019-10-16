import numpy as np
import pytest
import maszcal.check as check


def describe_emulation_errors():

    @pytest.fixture
    def emulation_errors():
        return check.EmulationErrors()

    def it_produces_a_number_versus_error_level_curve(emulation_errors):
        CON_MIN = 1
        CON_MAX = 2
        A_SZ_MIN = -1
        A_SZ_MAX = 1

        param_mins = np.array([CON_MIN, A_SZ_MIN])
        param_maxes = np.array([CON_MAX, A_SZ_MAX])

        errors, error_fracs = emulation_errors.get_baryonic_emulation_errors(
            param_mins,
            param_maxes,
            sampling='lh'
        )

        assert errors > 0
