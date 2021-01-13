from dataclasses import dataclass
import numpy as np


@dataclass
class WeakLensingData:
    radial_coordinates: np.ndarray
    redshifts: np.ndarray
    wl_signals: np.ndarray
    masses: np.ndarray

    def __post_init__(self):
        if not self._data_are_consistent():
            raise ValueError('wl_signals must have shape radial_coordinates.shape + redshifts.shape + (n,)')

    def _data_are_consistent(self):
        wl_radial_coordinates_and_redshifts_match = self.wl_signals.shape[:2] == self.radial_coordinates.shape + self.redshifts.shape
        wl_correct_dim = self.wl_signals.ndim >= 2
        masses_match_redshifts = self.masses.shape[0] == self.redshifts.size
        return wl_radial_coordinates_and_redshifts_match and wl_correct_dim and masses_match_redshifts

    def select_redshift_index(self, redshift_index):
        new_redshift = np.array([self.redshifts[redshift_index]])
        new_wl_signals = self.wl_signals[:, redshift_index:redshift_index+1, :]
        new_masses = self.masses[redshift_index:redshift_index+1, :]
        return WeakLensingData(
            radial_coordinates=self.radial_coordinates,
            redshifts=new_redshift,
            wl_signals=new_wl_signals,
            masses=new_masses,
        )


@dataclass
class StackedWeakLensingData:
    radial_coordinates: np.ndarray
    redshifts: np.ndarray
    stacked_wl_signal: np.ndarray
    masses: np.ndarray
