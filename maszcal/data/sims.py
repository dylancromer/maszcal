import numpy as np
from maszcal.data.templates import WeakLensingData


class NBatta2010(WeakLensingData):
    def __init__(self, data_dir='data/NBatta2010'):
        rs, wl_signals, zs = self._load_data(data_dir)
        super().__init__(
            radii=rs,
            redshifts=zs,
            wl_signals=wl_signals,
        )

    def _load_data(self, data_dir):
        return np.ones(10), np.ones(10), np.ones(10)
