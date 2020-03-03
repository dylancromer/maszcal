import numpy as np


class NfwSingleMass:
    def esd(self, rs, params):
        return np.ones((4, 1))

    def get_best_fit(self, data):
        return np.array([3, 1e14])
