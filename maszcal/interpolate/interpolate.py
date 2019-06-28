from dataclasses import dataclass
import numpy as np
from .rbf import Rbf
from maszcal.interp_utils import make_flat, combine_radii_with_params
from maszcal.nothing import NoSavedRbf




class RbfInterpolator:
    def __init__(self, rs, params, grid, saved_rbf=NoSavedRbf()):
        self.rs = rs
        self.params = params
        self.interp_grid = grid

        try:
            self.ndim = 1 + params.shape[1]
        except AttributeError:
            self.ndim = saved_rbf.dimension

        if not isinstance(saved_rbf, NoSavedRbf):
            self.rbfi = Rbf(saved_rbf=saved_rbf)

    def process(self, function='multiquadric'):
        point_coords = combine_radii_with_params(self.rs, self.params).T

        point_vals = make_flat(self.interp_grid)

        self.rbfi = Rbf(*point_coords, point_vals, function=function)

    def interp(self, rs, params):
        point_coords = combine_radii_with_params(rs, params).T

        n_rs = rs.size
        n_params = params.shape[0]

        try:
            return self.rbfi(*point_coords).reshape(n_rs, n_params)

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
