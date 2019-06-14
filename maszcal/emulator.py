import sys
import json
import numpy as np
import astropy.units as u
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.model import StackedModel
from maszcal.mathutils import atleast_kd
from maszcal.ioutils import NumpyEncoder
from maszcal.nothing import NoGrid, NoCoords, NoSavedRbf, NoInterpFile




class LargeErrorWarning(Warning):
    pass


class LensingEmulator:
    def __init__(self, comoving=True, units=u.Msun/u.Mpc**2):
        self.comoving = comoving
        self.units = units
        self.ERRCHECK_NUM = 2
        self.NORM = 1e12

    def init_stacked_model(self):
        self.stacked_model = StackedModel()
        self.stacked_model.comoving_radii = self.comoving

    def generate_grid(self, coords, units=u.Msun/u.Mpc**2):
        if len(coords) > 3:
            miscentered = True
        else:
            miscentered = False

        try:
            self.stacked_model.set_coords(coords)
            return (self.stacked_model.stacked_profile(miscentered=miscentered, units=self.units)
                    *atleast_kd(coords[0], len(coords))
                    /self.NORM)
        except AttributeError:
            self.init_stacked_model()
            self.stacked_model.set_coords(coords)
            return (self.stacked_model.stacked_profile(miscentered=miscentered, units=self.units)
                    *atleast_kd(coords[0], len(coords))
                    /self.NORM)

    def load_emulation(self, interpolation_file=NoInterpFile(), saved_rbf=NoSavedRbf()):
        if not isinstance(interpolation_file, NoInterpFile):
            saved_rbf = self.load_interpolation(interpolation_file)
            self.interpolator = RbfInterpolator(NoCoords(), NoGrid(), saved_rbf)
        elif not isinstance(saved_rbf, NoSavedRbf):
            self.interpolator = RbfInterpolator(NoCoords(), NoGrid(), saved_rbf)
        else:
            raise TypeError("load_emulation requires either an "
                            "interpolation file or a SavedRbf object")

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

    def load_interpolation(self, interp_file):
        with open(interp_file, 'r') as json_file:
            rbf_dict = json.load(json_file)

        for key,val in rbf_dict.items():
            if isinstance(val, list):
                rbf_dict[key] = np.asarray(val)

        return SavedRbf(**rbf_dict)

    def save_interpolation(self, interp_file=None):
        saved_rbf = self.interpolator.get_rbf_solution()

        if interp_file is None:
            return saved_rbf
        else:
            self._dump_saved_rbf(saved_rbf, interp_file)

    def _dump_saved_rbf(self, saved_rbf, rbf_file):
        rbf_dict = saved_rbf.__dict__
        with open(rbf_file, 'w') as outfile:
            json.dump(rbf_dict, outfile, cls=NumpyEncoder, ensure_ascii=False)

    def evaluate_on(self, coords):
        return self.interpolator.interp(coords)
