import json
import numpy as np
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
