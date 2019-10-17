from dataclasses import dataclass
import numpy as np
import astropy.units as u
import supercubos
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

    @staticmethod
    def _get_params(param_mins, param_maxes, num_samples, fixed_params, sampling):
        PARAM_AXIS = {'con':0, 'alpha':1, 'beta':2, 'gamma':3, 'a_sz':4}
        sampler = supercubos.LatinSampler()

        lh_params = np.zeros((num_samples, 5))
        if fixed_params is not None:
            fixed_param_axes = [PARAM_AXIS[param] for param in fixed_params.keys()]
            for param, val in fixed_params.items():
                lh_params[:, PARAM_AXIS[param]] = val
        else:
            fixed_param_axes = []

        free_param_mask = np.ones(5, dtype=bool)
        free_param_mask[fixed_param_axes] = False

        lh_params[:, free_param_mask] = sampler.get_lh_sample(param_mins, param_maxes, num_samples)
        return lh_params

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
            sampling='lh',
            fixed_params=None,
    ):
        params_to_interpolate = self._get_params(param_mins, param_maxes, num_samples, fixed_params, sampling)
        function_to_interpolate = self._get_function_to_interpolate(params_to_interpolate)

        emulator = self._get_emulator(params_to_interpolate, function_to_interpolate)

        test_params = self._get_test_params(params_to_interpolate)
        emulated_values = emulator.evaluate_on(test_params)
        true_values = self._get_true_values(test_params)

        rel_errors = np.abs((true_values - emulated_values)/true_values)
        error_levels = np.logspace(-8, 0, 100)
        error_fracs = np.array([self._percent_at_error_level(level, rel_errors) for level in error_levels])
        return error_levels, error_fracs
