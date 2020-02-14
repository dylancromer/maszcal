from dataclasses import dataclass
import numpy as np


@dataclass
class WeakLensingData:
    radii: np.ndarray
    redshifts: np.ndarray
    wl_signals: np.ndarray
