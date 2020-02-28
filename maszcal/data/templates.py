from dataclasses import dataclass
import numpy as np


@dataclass
class WeakLensingData:
    radii: np.ndarray
    redshifts: np.ndarray
    wl_signals: np.ndarray
    covariances: np.ndarray

    def __post_init__(self):
        if not self._data_are_consistent():
            raise ValueError('wl_signals must have shape radii.shape + redshifts.shape + (n,)')

    def _data_are_consistent(self):
        wl_radii_and_redshifts_match = self.wl_signals.shape[:2] == self.radii.shape + self.redshifts.shape
        wl_correct_dim = self.wl_signals.ndim == 3
        wl_matches_cov = self.wl_signals.shape[2] == self.covariances.shape[2]
        cov_matches_radii = self.covariances.shape[:2] == (self.radii.size, self.radii.size)

        return wl_radii_and_redshifts_match and wl_correct_dim and wl_matches_cov and cov_matches_radii

    def select_redshift_index(self, redshift_index):
        new_redshift = np.array([self.redshifts[redshift_index]])
        new_wl_signals = self.wl_signals[:, redshift_index:redshift_index+1, :]
        return WeakLensingData(
            radii=self.radii,
            redshifts=new_redshift,
            wl_signals=new_wl_signals,
            covariances=self.covariances,
        )
