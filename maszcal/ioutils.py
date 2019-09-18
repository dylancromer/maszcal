import json
import numpy as np
from scipy.interpolate import interp1d, interp2d
from maszcal.interpolate import SavedRbf


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class EmulationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SavedRbf):
            return obj.__dict__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_selection_func_interpolator(selection_func_file):
    with open(selection_func_file, 'r') as json_file:
        selec_func_dict = json.load(json_file)

    mus = np.asarray(selec_func_dict['mus'])
    zs = np.asarray(selec_func_dict['zs'])
    selection_fs = np.asarray(selec_func_dict['selection_fs'])
    interpolator = interp2d(zs, mus, selection_fs, kind='linear')

    return lambda mu, z: interpolator(z, mu)


def get_lensing_weights_interpolator(lensing_weights_file):
    with open(lensing_weights_file, 'r') as json_file:
        weights_dict = json.load(json_file)

    zs = np.asarray(weights_dict['zs'])
    weights = np.asarray(weights_dict['weights'])

    return interp1d(zs, weights, kind='cubic')
