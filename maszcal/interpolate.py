import numpy as np
import GPy as gpy
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
