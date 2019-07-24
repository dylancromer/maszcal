from dataclasses import dataclass
import numpy as np
from .rbf import Rbf
from maszcal.interp_utils import make_flat
import maszcal.nothing as nothing


class RbfInterpolator:
    def __init__(self, params, func_vals, saved_rbf=nothing.NoSavedRbf()):
        self.params = params
        self.interp_func_vals = func_vals

        try:
            self.ndim = params.shape[1]
        except AttributeError:
            self.ndim = saved_rbf.dimension

        if not isinstance(saved_rbf, nothing.NoSavedRbf):
            self.rbfi = Rbf(saved_rbf=saved_rbf)

    @classmethod
    def from_saved_rbf(cls, saved_rbf):
        return cls(nothing.NoCoords(), nothing.NoFuncVals(), saved_rbf=saved_rbf)

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
