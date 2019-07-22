import json
import numpy as np
from maszcal.interpolate import RbfInterpolator, SavedRbf
from maszcal.model import StackedModel
from maszcal.ioutils import NumpyEncoder
import maszcal.nothing as nothing




class LensingEmulator:
    def load_emulation(self, interpolation_file=nothing.NoInterpFile(), saved_rbf=nothing.NoSavedRbf()):

        if not isinstance(interpolation_file, nothing.NoInterpFile):
            saved_rbf = self._load_interpolation(interpolation_file)
            self.interpolator = RbfInterpolator(nothing.NoCoords(), nothing.NoFuncVals(), saved_rbf=saved_rbf)

        elif not isinstance(saved_rbf, nothing.NoSavedRbf):
            self.interpolator = RbfInterpolator(nothing.NoCoords(), nothing.NoFuncVals(), saved_rbf=saved_rbf)

        else:
            raise TypeError("load_emulation requires either an "
                            "interpolation file or a SavedRbf object")

    def emulate(self, params, func_vals):
        self.interpolator = RbfInterpolator(params, func_vals)
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

    def evaluate_on(self, params):
        return self.interpolator.interp(params)
