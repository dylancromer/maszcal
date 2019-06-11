from dataclasses import dataclass
import numpy as np
#import GPy as gpy
from scipy.interpolate import Rbf
from maszcal.interp_utils import cartesian_prod, make_flat




class NoKernel:
    pass


class GaussInterpolator:
    def __init__(self, coords, grid, kernel=NoKernel()):
        self.interp_coords = coords
        self.interp_grid = grid

        dim = len(coords)
        if isinstance(kernel, NoKernel):
            self.kernel = gpy.kern.RBF(dim)
        else:
            self.kernel = kernel

    def process(self):
        X = cartesian_prod(*self.interp_coords)
        Y = make_flat(self.interp_grid)[:, np.newaxis]

        self.model = gpy.models.GPRegression(X, Y, self.kernel)
        self.model.optimize()

    def interp(self, coords):
        x = cartesian_prod(*coords)

        try:
            y,y_err = self.model.predict(x)
            y = y.reshape(*(coord.size for coord in coords))
            y_err = y_err.reshape(*(coord.size for coord in coords))
            return y,y_err

        except AttributeError as err:
            raise AttributeError(str(err) + "\nGaussian process interpolation not yet optimized.\
                                  You must run GaussInterpolator.process() before \
                                  trying to evaluate the interpolator.")


class RbfInterpolator:
    def __init__(self, coords, grid):
        self.interp_coords = coords
        self.interp_grid = grid

    def process(self, function='multiquadric'):
        point_coords = cartesian_prod(*self.interp_coords).T
        point_vals = make_flat(self.interp_grid)
        self.rbfi = Rbf(*point_coords, point_vals, function=function)

    def interp(self, coords):
        point_coords = cartesian_prod(*coords).T
        try:
            return self.rbfi(*point_coords).reshape(*(coord.size for coord in coords))

        except AttributeError as err:
            raise AttributeError(str(err) + "\nRBF interpolation not yet calculated.\
                                  You must run RbfInterpolator.process() before \
                                  trying to evaluate the interpolator.")

    def get_rbf_solution(self):
        return SavedRbf(
            len(self.interp_coords),
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
