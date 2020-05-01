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
        self.params = params
        self.interp_func_vals = func_vals

        try:
            self.ndim = params.shape[1]
        except AttributeError:
            self.ndim = saved_rbf.dimension

        if not isinstance(saved_rbf, maszcal.nothing.NoSavedRbf):
            self.rbfi = Rbf(saved_rbf=saved_rbf)
        else:
            self.process()

    @classmethod
    def from_saved_rbf(cls, saved_rbf):
        return cls(maszcal.nothing.NoCoords(), maszcal.nothing.NoFuncVals(), saved_rbf=saved_rbf)

    def process(self):
        point_vals = make_flat(self.interp_func_vals)
        self.rbfi = Rbf(*self.params.T, point_vals, function=self.rbf_function)

    def __call__(self, params):
        try:
            return self.rbfi(*params.T)

        except AttributeError as err:
            raise AttributeError(str(err) + "\nRBF interpolation not yet calculated.\
                                  You must run RbfInterpolator.process() before \
                                  trying to evaluate the interpolator.")

    def get_rbf_solution(self):
        return SavedRbf(
            dimension=self.ndim,
            norm=self.rbfi.norm,
            function=self.rbfi.function,
            data=self.rbfi.di,
            coords=self.rbfi.xi,
            epsilon=self.rbfi.epsilon,
            smoothness=self.rbfi.smooth,
            nodes=self.rbfi.nodes,
        )


class RbfMismatchError(Exception):
    pass


@dataclass
class SavedRbf:
    dimension: int
    norm: str
    function: str
    data: np.ndarray
    coords: np.ndarray
    epsilon: float
    smoothness: float
    nodes: np.ndarray

    def __post_init__(self):
        if self.data.size != self.nodes.size:
            raise RbfMismatchError("Node and data are not the same size")
        if self.coords.size != self.data.size*self.dimension:
            raise RbfMismatchError("Data, coords, and dimension are inconsistent")
        if not isinstance(self.dimension, int):
            raise TypeError("Dimension must be an integer")
