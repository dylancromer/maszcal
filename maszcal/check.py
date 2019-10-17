from dataclasses import dataclass
import numpy as np
import astropy.units as u
import maszcal.defaults as defaults
import maszcal.emulator


@dataclass
class BaryonicEmulationErrors:
    radii: np.ndarray
    log_masses: np.ndarray
    redshifts: np.ndarray
    units: u.Quantity = u.Msun/u.pc**2
    comoving: bool = True
    delta: int = 200
    mass_definition: str = 'mean'
    cosmo_params: object = defaults.DefaultCosmology()
    selection_func_file: object = defaults.DefaultSelectionFunc()
    lensing_weights_file: object = defaults.DefaultLensingWeights()
    emulator_class: object = maszcal.emulator.LensingEmulator

    def _get_params_from_partial(self, param_mins, param_maxes, fixed_params, sampling):
        pass

    def _get_function_to_interpolate(self, params_to_interpolate):
        pass

    def _get_emulator(self, params_to_interpolate, function_to_interpolate):
        emulator = self.emulator_class()
        emulator.emulate(self.radii, params_to_interpolate, function_to_interpolate)
        return emulator

    def _get_test_params(self, params_to_interpolate):
        pass

    def _get_true_values(self, test_params):
        return 2*np.ones(2)

    def _percent_at_error_level(self, error_level, errors):
        errors_above_level = errors[np.where(np.abs(errors) > error_level)]
        return 100*(errors_above_level.size/errors.size)

    def get_emulation_errors(
            self,
            param_mins,
            param_maxes,
            num_samples,
            sampling='all',
            fixed_params=None,
    ):
        params_to_interpolate = self._get_params_from_partial(param_mins, param_maxes, fixed_params, sampling)
        function_to_interpolate = self._get_function_to_interpolate(params_to_interpolate)

        emulator = self._get_emulator(params_to_interpolate, function_to_interpolate)

        test_params = self._get_test_params(params_to_interpolate)
        emulated_values = emulator.evaluate_on(test_params)
        true_values = self._get_true_values(test_params)

        rel_errors = np.abs((true_values - emulated_values)/true_values)
        error_levels = np.logspace(-8, 0, 100)
        error_fracs = np.array([self._percent_at_error_level(level, rel_errors) for level in error_levels])
        return error_levels, error_fracs
