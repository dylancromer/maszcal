from dataclasses import dataclass
import numpy as np
from .rbf import Rbf
from maszcal.interp_utils import make_flat, combine_radii_with_params
from maszcal.nothing import NoSavedRbf




class RbfInterpolator:
    def __init__(self, params, func_vals, saved_rbf=NoSavedRbf()):
        self.params = params
        self.interp_func_vals = func_vals

        try:
            self.ndim = params.shape[1]
        except AttributeError:
            self.ndim = saved_rbf.dimension

        if not isinstance(saved_rbf, NoSavedRbf):
            self.rbfi = Rbf(saved_rbf=saved_rbf)

    def process(self, function='multiquadric'):
        point_vals = make_flat(self.interp_func_vals)

        self.rbfi = Rbf(*self.params.T, point_vals, function=function)

    def interp(self, params):
        try:
            return self.rbfi(*params.T)

        except AttributeError as err:
            raise AttributeError(str(err) + "\nRBF interpolation not yet calculated.\
                                  You must run RbfInterpolator.process() before \
                                  trying to evaluate the interpolator.")

    def get_rbf_solution(self):
        return SavedRbf(
            self.ndim,
            self.rbfi.norm,
            self.rbfi.function,
            self.rbfi.di,
            self.rbfi.xi,
            self.rbfi.epsilon,
            self.rbfi.smooth,
            self.rbfi.nodes,
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
