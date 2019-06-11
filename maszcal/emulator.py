import sys
import json
import numpy as np
from maszcal.interpolate import RbfInterpolator
from maszcal.model import StackedModel
from maszcal.mathutils import atleast_kd
from maszcal.ioutils import NumpyEncoder




class LargeErrorWarning(Warning):
    pass


class NoGrid:
    pass


class LensingEmulator:
    def __init__(self, comoving=True):
        self.comoving = comoving
        self.ERRCHECK_NUM = 2
        self.NORM = 1e12

    def init_stacked_model(self):
        self.stacked_model = StackedModel()
        self.stacked_model.comoving_radii = self.comoving

    def generate_grid(self, coords):
        if len(coords) > 3:
            miscentered = True
        else:
            miscentered = False

        try:
            self.stacked_model.set_coords(coords)
            return (self.stacked_model.stacked_profile(miscentered=miscentered)
                    *atleast_kd(coords[0], len(coords))
                    /self.NORM)

        except AttributeError:
            self.init_stacked_model()
            self.stacked_model.set_coords(coords)
            return (self.stacked_model.stacked_profile(miscentered=miscentered)
                    *atleast_kd(coords[0], len(coords))
                    /self.NORM)

    def emulate(self, coords, grid=NoGrid(), check_errs=False):
        if isinstance(grid, NoGrid):
            grid = self.generate_grid(coords)

        self.interpolator = RbfInterpolator(coords, grid)
        self.interpolator.process()

        if check_errs:
            errcheck_size = self.ERRCHECK_NUM**len(coords)
            if errcheck_size >  1000:
                raise ValueError("Error checking using too many samples: will"
                                 f" result in {errcheck_size} total samples")
                sys.exit()

            self.check_errors(coords)

    def check_errors(self, coords):
        rand_coords = [coords[0]]
        for coord in coords[1:]:
            coord_length = coord.max() - coord.min()
            min_ = coord.min()
            rand_coord = coord_length*np.random.rand(self.ERRCHECK_NUM) + min_
            rand_coords.append(rand_coord)

        interp_values = self.evaluate_on(rand_coords)
        true_values = self.generate_grid(rand_coords)

        if not np.allclose(interp_values, true_values, rtol=1e-1):
            raise LargeErrorWarning("Some interpolation errors exceed 10%")

        rel_err_mean = np.abs((interp_values - true_values)/true_values).mean()

        if rel_err_mean > 1e-2:
            raise LargeErrorWarning("Mean error of the interpolation exceeds 1%")

    def save_interpolation(self):
        saved_rbf = self.interpolator.get_rbf_solution()
        self._dump_saved_rbf(saved_rbf)

    def _dump_saved_rbf(self, saved_rbf):
        rbf_dict = {
            'norm':saved_rbf.norm,
            'function':saved_rbf.function,
            'data':saved_rbf.data,
            'coords':saved_rbf.coords,
            'epsilon':saved_rbf.epsilon,
            'smoothness':saved_rbf.smoothness,
            'nodes':saved_rbf.nodes,
        }

        json_dump = json.dumps(rbf_dict, cls=NumpyEncoder)
        assert False, print(json_dump)

    def evaluate_on(self, coords):
        return self.interpolator.interp(coords)
