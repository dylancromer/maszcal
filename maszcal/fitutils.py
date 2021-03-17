import numpy as np
import scipy.optimize


def _diffy_evo(func, bounds):
    return scipy.optimize.differential_evolution(func, bounds)


def _select_global_backend(method_name):
    available_methods = {
        'global-differential-evolution': _diffy_evo,
    }

    try:
        return available_methods[method_name]
    except KeyError:
        raise ValueError(f'{method_name} is not an available minimization method')


def global_minimize(func, param_mins, param_maxes, method):
    bounds = np.stack((param_mins, param_maxes), axis=1)
    _minimize = _select_global_backend(method)
    return _minimize(func, bounds).x
