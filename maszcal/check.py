from dataclasses import dataclass
import numpy as np
import astropy.units as u
import supercubos
import maszcal.defaults as defaults
import maszcal.emulator
import maszcal.lensing


@dataclass
class BaryonicEmulationErrors:
    radii: np.ndarray
    log_masses: np.ndarray
    redshifts: np.ndarray
    num_test_samples: int
    units: u.Quantity = u.Msun/u.pc**2
    comoving: bool = True
    delta: int = 200
    mass_definition: str = 'mean'
    cosmo_params: object = defaults.DefaultCosmology()
    selection_func_file: object = defaults.DefaultSelectionFunc()
    lensing_weights_file: object = defaults.DefaultLensingWeights()
    emulator_class: object = maszcal.emulator.LensingEmulator
    lensing_signal_class: object = maszcal.lensing.StackedBaryonLensingSignal

    @classmethod
    def _get_sampled_params(cls, param_mins, param_maxes, num_samples, sampling_method):
        sampler = supercubos.LatinSampler()
        sampling_funcs = {
            'lh':sampler.get_lh_sample,
            'sym':sampler.get_sym_sample,
            'rand':sampler.get_rand_sample,
        }

        sampler_func = sampling_funcs[sampling_method]

        return sampler_func(param_mins, param_maxes, num_samples)

    @classmethod
    def _get_params(cls, param_mins, param_maxes, num_samples, fixed_params, sampling_method):
        PARAM_AXIS = {'con':0, 'alpha':1, 'beta':2, 'gamma':3, 'a_sz':4}

        params = np.zeros((num_samples, 5))
        if fixed_params is not None:
            fixed_param_axes = [PARAM_AXIS[param] for param in fixed_params.keys()]
            for param, val in fixed_params.items():
                params[:, PARAM_AXIS[param]] = val
        else:
            fixed_param_axes = []

        free_param_mask = np.ones(5, dtype=bool)
        free_param_mask[fixed_param_axes] = False

        params[:, free_param_mask] = cls._get_sampled_params(
            param_mins,
            param_maxes,
            num_samples,
            sampling_method
        )

        return params

    def _get_lensing_signal(self):
        return self.lensing_signal_class(
            log_masses=self.log_masses,
            redshifts=self.redshifts,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            comoving=self.comoving,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def _get_function_values(self, params_to_interpolate):
        lensing_signal = self._get_lensing_signal()

        return self.radii[:, None] * lensing_signal.stacked_esd(self.radii, params_to_interpolate)

    def _get_emulator(self, params_to_interpolate, function_to_interpolate):
        emulator = self.emulator_class()
        emulator.emulate(self.radii, params_to_interpolate, function_to_interpolate)
        return emulator

    def _get_test_params(self, param_mins, param_maxes, fixed_params):
        return self._get_params(param_mins, param_maxes, self.num_test_samples, fixed_params, 'rand')

    def _percent_at_error_level(self, error_level, errors):
        errors_above_level = errors[np.where(np.abs(errors) > error_level)]
        return 100*(errors_above_level.size/errors.size)

    def _check_param_limits(self, param_mins, param_maxes):
        if param_mins.shape != param_maxes.shape:
            raise ValueError('param_mins and param_maxes must both be of same length.')

    def get_emulation_errors(
            self,
            param_mins,
            param_maxes,
            num_samples,
            sampling_method='lh',
            fixed_params=None,
    ):
        self._check_param_limits(param_mins, param_maxes)

        params_to_interpolate = self._get_params(param_mins, param_maxes, num_samples, fixed_params, sampling_method)
        function_to_interpolate = self._get_function_values(params_to_interpolate)

        emulator = self._get_emulator(params_to_interpolate, function_to_interpolate)

        test_params = self._get_test_params(param_mins, param_maxes, fixed_params)
        emulated_values = emulator.evaluate_on(test_params)
        true_values = self._get_function_values(test_params)

        rel_errors = np.abs((true_values - emulated_values)/true_values)
        error_levels = np.logspace(-8, 0, 100)
        error_fracs = np.array([self._percent_at_error_level(level, rel_errors) for level in error_levels])
        return error_levels, error_fracs
