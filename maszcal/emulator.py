import json
from dataclasses import dataclass
import numpy as np
from maszcal.interpolate import RbfInterpolator, SavedRbf
import maszcal.ioutils as ioutils
import maszcal.nothing as nothing


class LensingEmulator:
    def load_emulation(
            self,
            emulation_file=nothing.NoFile(),
            saved_emulation=nothing.NoEmulation()
    ):

        if not isinstance(emulation_file, nothing.NoFile):
            saved_emulation = Emulation.load(emulation_file)
            self.radii = saved_emulation.radii
            self.interpolators = self._init_saved_rbfs(saved_emulation.saved_rbfs)

        elif not isinstance(saved_emulation, nothing.NoEmulation):
            self.radii = saved_emulation.radii
            self.interpolators = self._init_saved_rbfs(saved_emulation.saved_rbfs)

        else:
            raise TypeError("load_emulation requires either an "
                            "interpolation file or a SavedRbf object")

    def _init_saved_rbfs(self, saved_rbfs):
        return [RbfInterpolator.from_saved_rbf(saved_rbf) for saved_rbf in saved_rbfs]

    def _get_saved_rbfs(self):
        return [interp.get_rbf_solution() for interp in self.interpolators]

    def save_emulation(self, emulation_file=None):
        emulation = Emulation(
            radii=self.radii,
            saved_rbfs=self._get_saved_rbfs(),
        )

        if emulation_file is None:
            return emulation
        else:
            emulation.save(emulation_file)

    def emulate(self, rs, params, func_vals):
        self.radii = rs
        self.interpolators = [self._emulate_single_radius(params, val) for val in func_vals]

    def _emulate_single_radius(self, params, func_vals):
        interpolator = RbfInterpolator(params, func_vals)
        interpolator.process()
        return interpolator

    def evaluate_on(self, params):
        return np.array([self.interpolators[i].interp(params) for i in range(self.radii.size)])


@dataclass
class Emulation:
    radii: np.ndarray
    saved_rbfs: list

    def __post_init__(self):
        if self.radii.ndim > 1:
            raise TypeError("radii must be a 1-dimensional array")
        if self.radii.size != len(self.saved_rbfs):
            raise ValueError("radii and saved_rbfs must have equal length")

    def save(self, save_location):
        emulation_dict = self.__dict__
        with open(save_location, 'w') as outfile:
            json.dump(emulation_dict, outfile, cls=ioutils.EmulationEncoder, ensure_ascii=False)

    @staticmethod
    def _init_saved_rbf(rbf_dict):
        for key, val in rbf_dict.items():
            if isinstance(val, list):
                rbf_dict[key] = np.asarray(val)
        return SavedRbf(**rbf_dict)

    @classmethod
    def load(cls, saved_emulation):
        with open(saved_emulation, 'r') as json_file:
            emulation_dict = json.load(json_file)

        saved_rbf_dicts = emulation_dict['saved_rbfs']
        emulation_dict['saved_rbfs'] = [cls._init_saved_rbf(rbf_dict) for rbf_dict in saved_rbf_dicts]

        emulation_dict['radii'] = np.asarray(emulation_dict['radii'])

        return cls(**emulation_dict)
