from dataclasses import dataclass
import numpy as np
import pytest
import maszcal.check as check
from maszcal.cosmology import CosmoParams


@dataclass
class PretendEmulator:
    function: str = 'multiquadric'
    def emulate(self, radii, params, func_vals):
        self.radii = radii

    def evaluate_on(self, params):
        return np.ones((self.radii.size, params.shape[0]))


class PretendLensingSignal:
    def __init__(
            self,
            log_masses=None,
            redshifts=None,
            units=1,
            comoving=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=None,
            selection_func_file=None,
            lensing_weights_file=None,
    ):
        pass

    def stacked_esd(self, rs, params):
        return np.ones((rs.size, params.shape[0]))


def describe_emulation_errors():

    def describe__get_params():

        @pytest.fixture
        def params_fixed_on_axis_0():
            param_mins = np.array([0, 0, 0, 0])
            param_maxes = np.array([1, 1, 1, 1])

            num_samples = 10

            fixed_params = {'con': 3}

            return check.EmulationErrors._get_params(
                {'con':0, 'alpha':1, 'beta':2, 'gamma':3, 'a_sz':4},
                param_mins,
                param_maxes,
                num_samples,
                fixed_params=fixed_params,
                sampling_method='lh'
            )

        def it_returns_params_with_a_constant_column_if_a_param_is_fixed(params_fixed_on_axis_0):
            assert np.all(params_fixed_on_axis_0[:, 0] == params_fixed_on_axis_0[0, 0])

        def it_returns_params_not_constant_if_the_param_is_not_fixed(params_fixed_on_axis_0):
            assert not np.all(params_fixed_on_axis_0[:, 1] == params_fixed_on_axis_0[0, 1])

    def describe_get_emulation_errors():

        @pytest.fixture
        def emulation_errors():
            rs = np.logspace(-1, 1, 5)
            mus = np.linspace(np.log(1e14), np.log(1e16), 6)
            zs = np.linspace(0, 1, 6)

            emulator_class = PretendEmulator
            lensing_signal_class = PretendLensingSignal
            return check.EmulationErrors(
                lensing_signal_class=lensing_signal_class,
                lensing_param_axes={'con':0, 'alpha':1, 'beta':2, 'gamma':3, 'a_sz':4},
                radii=rs,
                log_masses=mus,
                redshifts=zs,
                num_test_samples=10,
                emulator_class=emulator_class,
            )

        def it_produces_an_error_percent_curve_that_is_monotonically_decreasing(emulation_errors):
            CON_MIN = 1
            CON_MAX = 2
            A_SZ_MIN = -1
            A_SZ_MAX = 1

            param_mins = np.array([CON_MIN, A_SZ_MIN])
            param_maxes = np.array([CON_MAX, A_SZ_MAX])

            fixed_params = {'alpha': 0.88, 'beta': 3.8, 'gamma': 0.2}

            num_samples = 10

            error_levels, error_fracs = emulation_errors.get_emulation_errors(
                param_mins,
                param_maxes,
                num_samples,
                sampling_method='lh',
                fixed_params=fixed_params,
            )

            assert np.all(error_fracs[1:] <= error_fracs[:-1])

    def describe_input_handling():

        @pytest.fixture
        def emulation_errors():
            rs = np.logspace(-1, 1, 5)
            mus = np.linspace(np.log(1e14), np.log(1e16), 6)
            zs = np.linspace(0, 1, 6)

            emulator_class = PretendEmulator
            lensing_signal_class = PretendLensingSignal
            return check.EmulationErrors(
                lensing_signal_class=lensing_signal_class,
                lensing_param_axes={'con':0, 'alpha':1, 'beta':2, 'gamma':3, 'a_sz':4},
                radii=rs,
                log_masses=mus,
                redshifts=zs,
                num_test_samples=10,
                emulator_class=emulator_class,
            )

        def it_errors_if_you_have_mismatching_params_limits(emulation_errors):
            param_mins = np.array([0, 0, 0, 0])
            param_maxes = np.array([1, 1, 1, 1, 1])

            with pytest.raises(ValueError):
                emulation_errors._check_param_limits(param_mins, param_maxes)

        def it_errors_if_fixed_plus_free_params_arent_5(emulation_errors):
            with pytest.raises(ValueError):
                emulation_errors._check_num_of_params(param_limits=np.array([0, 0, 0]), fixed_params=None)

            with pytest.raises(ValueError):
                emulation_errors._check_num_of_params(param_limits=np.array([0, 0, 0]), fixed_params={'con': 3})

        def it_errors_if_your_sampling_method_is_wrong(emulation_errors):
            with pytest.raises(ValueError):
                emulation_errors._check_sampling_method('enrfienfie')
