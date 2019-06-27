import sys
import json
import numpy as np
import astropy.units as u
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.model import StackedModel
from maszcal.mathutils import atleast_kd
from maszcal.ioutils import NumpyEncoder
from maszcal.nothing import NoGrid, NoCoords, NoSavedRbf, NoInterpFile




class LensingEmulator:
    def __init__(self, comoving=True, units=u.Msun/u.Mpc**2):
        self.comoving = comoving
        self.units = units

    def init_stacked_model(self):
        self.stacked_model = StackedModel()
        self.stacked_model.comoving_radii = self.comoving

    def load_emulation(self, interpolation_file=NoInterpFile(), saved_rbf=NoSavedRbf()):
        if not isinstance(interpolation_file, NoInterpFile):
            saved_rbf = self._load_interpolation(interpolation_file)
            self.interpolator = RbfInterpolator(NoCoords(), NoGrid(), saved_rbf=saved_rbf)
        elif not isinstance(saved_rbf, NoSavedRbf):
            self.interpolator = RbfInterpolator(NoCoords(), NoGrid(), saved_rbf=saved_rbf)
        else:
            raise TypeError("load_emulation requires either an "
                            "interpolation file or a SavedRbf object")

    def emulate(self, coords, grid, coords_separated=True):
        self.interpolator = RbfInterpolator(coords, grid, coords_separated=coords_separated)
        self.interpolator.process()

    def _load_interpolation(self, interp_file):
        with open(interp_file, 'r') as json_file:
            rbf_dict = json.load(json_file)

        for key,val in rbf_dict.items():
            if isinstance(val, list):
                rbf_dict[key] = np.asarray(val)

        return SavedRbf(**rbf_dict)

    def save_emulation(self, interp_file=None):
        saved_rbf = self.interpolator.get_rbf_solution()

        if interp_file is None:
            return saved_rbf
        else:
            self._dump_saved_rbf(saved_rbf, interp_file)

    def _dump_saved_rbf(self, saved_rbf, rbf_file):
        rbf_dict = saved_rbf.__dict__
        with open(rbf_file, 'w') as outfile:
            json.dump(rbf_dict, outfile, cls=NumpyEncoder, ensure_ascii=False)

    def evaluate_on(self, coords, coords_separated=True):
        return self.interpolator.interp(coords, coords_separated=coords_separated)
