import numpy as np
import scipy.optimize


def _select_optimize_function(method_name):
    available_methods = {
        'global-differential-evolution': scipy.optimize.differential_evolution,
        'local-nelder-mead': lambda func, guess: scipy.optimize.minimize(
            func,
            guess,
            method='Nelder-Mead',
        ),
    }

    try:
        return available_methods[method_name]
    except KeyError:
        raise ValueError(f'{method_name} is not an available minimization method')


def minimize(func, param_mins, param_maxes, method):
    bounds = np.stack((param_mins, param_maxes), axis=1)
    guess = (param_maxes + param_mins)/2

    if 'local' in method:
        arg = guess
    elif 'global' in method:
        arg = bounds
    else:
        raise ValueError(f'{method} is neither local nor global')

    _minimize = _select_optimize_function(method)

    return _minimize(func, arg).x
