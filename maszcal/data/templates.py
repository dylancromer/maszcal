from dataclasses import dataclass
import numpy as np


@dataclass
class WeakLensingData:
    radii: np.ndarray
    redshifts: np.ndarray
    wl_signals: np.ndarray

    def __post_init__(self):
        if not self._data_are_consistent(self.radii, self.redshifts, self.wl_signals):
            raise ValueError('wl_signals must have shape radii.shape + redshifts.shape + (n,)')

    def _data_are_consistent(self, radii, redshifts, wl_signals):
        condition_1 = wl_signals.shape[:2] == radii.shape + redshifts.shape
        condition_2 = wl_signals.ndim <= 3
        return condition_1 and condition_2
