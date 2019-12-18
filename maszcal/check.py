from dataclasses import dataclass
import numpy as np
import astropy.units as u
import supercubos
import maszcal.defaults as defaults
import maszcal.emulator
import maszcal.lensing


@dataclass
class EmulationErrors:
    lensing_signal_class: object
    lensing_param_axes: dict
    radii: np.ndarray
    log_masses: np.ndarray
    redshifts: np.ndarray
    num_test_samples: int
    units: u.Quantity = u.Msun/u.pc**2
    comoving: bool = True
    delta: int = 200
    mass_definition: str = 'mean'
    rbf_function: str = 'multiquadric'
    cosmo_params: object = defaults.DefaultCosmology()
    selection_func_file: object = defaults.DefaultSelectionFunc()
    lensing_weights_file: object = defaults.DefaultLensingWeights()
    emulator_class: object = maszcal.emulator.LensingEmulator

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
    def _get_params(cls, lensing_param_axes, param_mins, param_maxes, num_samples, fixed_params, sampling_method):
        num_params = len(lensing_param_axes)
        params = np.zeros((num_samples, num_params))
        if fixed_params is not None:
            fixed_param_axes = [lensing_param_axes[param] for param in fixed_params.keys()]
            for param, val in fixed_params.items():
                params[:, lensing_param_axes[param]] = val
        else:
            fixed_param_axes = []

        free_param_mask = np.ones(num_params, dtype=bool)
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
        emulator = self.emulator_class(function=self.rbf_function)
        emulator.emulate(self.radii, params_to_interpolate, function_to_interpolate)
        return emulator

    def _get_test_params(self, param_mins, param_maxes, fixed_params):
        return self._get_params(self.lensing_param_axes, param_mins, param_maxes, self.num_test_samples, fixed_params, 'rand')

    def _percent_at_error_level(self, error_level, errors):
        errors_above_level = errors[np.where(np.abs(errors) > error_level)]
        return 100*(errors_above_level.size/errors.size)

    def _check_param_limits(self, param_mins, param_maxes):
        if param_mins.size != param_maxes.size:
            raise ValueError('param_mins and param_maxes must both be of same size.')

    def _check_num_of_params(self, param_limits, fixed_params):
        if fixed_params is None:
            num_of_fixed_params = 0
        else:
            num_of_fixed_params = len(fixed_params)

        if param_limits.size + num_of_fixed_params != len(self.lensing_param_axes):
            raise ValueError('Total number of parameters incorrect: '\
                             'length of param_mins/param_maxes and '\
                             'length of fixed_params must sum to the length'\
                             'of lensing_param_axes')

    def _check_sampling_method(self, sampling_method):
        if sampling_method not in ['lh', 'sym', 'rand']:
            raise ValueError('sampling_method must be \'lh\', \'sym\', or \'rand\'')

    def get_emulation_errors(
            self,
            param_mins,
            param_maxes,
            num_samples,
            sampling_method='lh',
            fixed_params=None,
    ):
        self._check_param_limits(param_mins, param_maxes)
        self._check_num_of_params(param_mins, fixed_params)
        self._check_sampling_method(sampling_method)

        params_to_interpolate = self._get_params(self.lensing_param_axes, param_mins, param_maxes, num_samples, fixed_params, sampling_method)
        function_to_interpolate = self._get_function_values(params_to_interpolate)

        emulator = self._get_emulator(params_to_interpolate, function_to_interpolate)

        test_params = self._get_test_params(param_mins, param_maxes, fixed_params)
        emulated_values = emulator.evaluate_on(test_params)
        true_values = self._get_function_values(test_params)

        rel_errors = np.abs((true_values - emulated_values)/true_values)
        error_levels = np.logspace(-8, 0, 100)
        error_fracs = np.array([self._percent_at_error_level(level, rel_errors) for level in error_levels])
        return error_levels, error_fracs
