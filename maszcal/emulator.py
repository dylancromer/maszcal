import numpy as np
from maszcal.interpolate import RbfInterpolator
from maszcal.model import StackedModel
from maszcal.mathutils import atleast_kd




class LargeErrorWarning(Warning):
    pass


class NoGrid:
    pass


class LensingEmulator:
    def __init__(self, comoving=True):
        self.comoving = comoving
        self.ERRCHECK_NUM = 100
        self.NORM = 1e12

    def init_stacked_model(self):
        self.stacked_model = StackedModel()
        self.stacked_model.comoving_radii = self.comoving

    def generate_grid(self, coords):
        try:
            self.stacked_model.set_coords(coords)
            return self.stacked_model.stacked_profile()*coords[0][:, None, None]/self.NORM
        except AttributeError:
            self.init_stacked_model()
            self.stacked_model.set_coords(coords)
            return self.stacked_model.stacked_profile()*coords[0][:, None, None]/self.NORM

    def emulate(self, coords, grid=NoGrid(), check_errs=True):
        if isinstance(grid, NoGrid):
            grid = self.generate_grid(coords)

        self.interpolator = RbfInterpolator(coords, grid)
        self.interpolator.process()

        if check_errs:
            self.check_errors(coords)

    def check_errors(self, coords):
        rand_coords = []
        for coord in coords:
            coord_length = coord.max() - coord.min()
            min_ = coord.min()
            rand_coord = coord_length*np.random.rand(self.ERRCHECK_NUM) + min_
            rand_coords.append(rand_coord)

        interp_values = self.interpolator.interp(rand_coords)
        true_values = self.generate_grid(rand_coords)*atleast_kd(rand_coords[0], interp_values.ndim)/self.NORM

        rel_err_mean = np.abs((interp_values - true_values)/true_values).mean()

        if rel_err_mean > 1e-2:
            raise LargeErrorWarning("Error of the interpolation exceeds 1%")

    def evaluate_on(self, coords):
        return self.interpolator.interp(coords)
