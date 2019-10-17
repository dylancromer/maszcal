import numpy as np
import pytest
import maszcal.check as check
from maszcal.cosmology import CosmoParams


class PretendEmulator:
    def emulate(self, radii, params, func_vals):
        pass

    def evaluate_on(self, params):
        return np.ones(2)


def describe_emulation_errors():

    def describe__get_params_from_partial():

        def it_gets_a_parameter_sample():
            param_mins = np.array([0, 0, 0, 0, 0])
            param_maxes = np.array([1, 1, 1, 1, 1])

            num_samples = 10

            params = check.BaryonicEmulationErrors._get_params(
                param_mins,
                param_maxes,
                num_samples,
                fixed_params=None,
                sampling='lh'
            )
            assert np.all(params > 0)

        def it_returns_params_with_a_constant_column_if_a_param_is_fixed():
            param_mins = np.array([0, 0, 0, 0])
            param_maxes = np.array([1, 1, 1, 1])

            num_samples = 10

            fixed_params = {'con':3}

            params = check.BaryonicEmulationErrors._get_params(
                param_mins,
                param_maxes,
                num_samples,
                fixed_params=fixed_params,
                sampling='lh'
            )

            assert params.shape == (10, 5)
            assert np.all(params[:, 0] == params[0, 0])

    def describe_get_emulation_errors():

        @pytest.fixture
        def emulation_errors():
            rs = np.logspace(-1, 1, 5)
            mus = np.linspace(np.log(1e14), np.log(1e16), 6)
            zs = np.linspace(0, 1, 6)

            emulator_class = PretendEmulator
            return check.BaryonicEmulationErrors(rs, mus, zs, emulator_class=emulator_class)

        def it_produces_an_error_percent_curve_that_is_monotonically_decreasing(emulation_errors):
            CON_MIN = 1
            CON_MAX = 2
            A_SZ_MIN = -1
            A_SZ_MAX = 1

            param_mins = np.array([CON_MIN, A_SZ_MIN])
            param_maxes = np.array([CON_MAX, A_SZ_MAX])

            fixed_params = {'alpha':0.88, 'beta':3.8, 'gamma':0.2}

            num_samples = 10

            error_levels, error_fracs = emulation_errors.get_emulation_errors(
                param_mins,
                param_maxes,
                num_samples,
                sampling='lh',
                fixed_params=fixed_params,
            )

            assert np.all(error_fracs[1:] <= error_fracs[:-1])

    def describe_init():

        def it_requires_redshifts():
            rs = np.logspace(-1, 1, 5)
            mus = np.linspace(np.log(1e14), np.log(1e16), 6)
            zs = np.linspace(0, 1, 6)
            with pytest.raises(TypeError):
                check.BaryonicEmulationErrors(radii=rs, log_masses=mus)

        def it_accepts_a_selection_func_file(mocker):
            rs = np.logspace(-1, 1, 5)
            mus = np.ones(10)
            zs = np.ones(5)
            sel_func_file = 'test/file/here'
            emu_errs = check.BaryonicEmulationErrors(rs, mus, zs, selection_func_file=sel_func_file)

            assert emu_errs.selection_func_file == sel_func_file

        def it_accepts_a_weights_file(mocker):
            rs = np.logspace(-1, 1, 5)
            mus = np.ones(10)
            zs = np.ones(5)
            weights_file = 'test/file/here'
            emu_errs = check.BaryonicEmulationErrors(rs, mus, zs, lensing_weights_file=weights_file)

            assert emu_errs.lensing_weights_file == weights_file

        def it_allows_a_different_mass_definition(mocker):
            rs = np.logspace(-1, 1, 5)
            mus = np.ones(10)
            zs = np.ones(5)

            delta = 500
            mass_definition = 'crit'

            emu_errs = check.BaryonicEmulationErrors(rs, mus, zs, delta=delta, mass_definition=mass_definition)
            assert emu_errs.mass_definition == mass_definition

        def it_can_use_a_different_cosmology(mocker):
            rs = np.logspace(-1, 1, 5)
            mus = np.ones(10)
            zs = np.ones(5)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            emu_errs = check.BaryonicEmulationErrors(rs, mus, zs, cosmo_params=cosmo)

            assert emu_errs.cosmo_params == cosmo
