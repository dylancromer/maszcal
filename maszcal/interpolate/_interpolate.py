from dataclasses import dataclass
import numpy as np
import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import smolyak
from .rbf import Rbf
from maszcal.interp_utils import make_flat
import maszcal.mathutils
import maszcal.nothing


class GaussianProcessInterpolator:
    def __init__(
            self,
            params,
            func_vals,
            kernel=sklearn.gaussian_process.kernels.RBF(),
    ):
        params = maszcal.mathutils.atleast_kd(params, 2)
        self.gpi = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(params, func_vals)

    def __call__(self, params):
        params = maszcal.mathutils.atleast_kd(params, 2)
        return self.gpi.predict(params)


class SmolyakInterpolator:
    def __init__(self, smolyak_grid, func_vals):
        self._smolyak_interpolator = smolyak.interp.SmolyakInterp(smolyak_grid, func_vals)

    def __call__(self, smolyak_grid):
        return self._smolyak_interpolator.interpolate(smolyak_grid)


class RbfInterpolator:
    rbf_function = 'multiquadric'

    def __init__(self, params, func_vals, saved_rbf=maszcal.nothing.NoSavedRbf()):
        params = maszcal.mathutils.atleast_kd(params, 2)
        point_vals = make_flat(func_vals)
        self.rbfi = Rbf(*params.T, point_vals, function=self.rbf_function)

    def __call__(self, params):
        params = maszcal.mathutils.atleast_kd(params, 2)
        return self.rbfi(*params.T)
