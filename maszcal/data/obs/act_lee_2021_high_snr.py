import numpy as np
import maszcal.cosmology
from maszcal.data.templates import StackedWeakLensingData


class ActLee2021HighSnr(StackedWeakLensingData):
    FROM_ARCMIN = 2 * np.pi / 360 / 60

    def __init__(self, data_dir='data/act-eunseong/high-snr/'):
        self.cosmology = self._init_cosmology()

        thetas, wl_signals, zs, masses = self._load_data(data_dir)

        super().__init__(
            radial_coordinates=thetas,
            redshifts=zs,
            stacked_wl_signal=wl_signals,
            masses=masses,
        )

    def _init_cosmology(self):
        return maszcal.cosmology.CosmoParams()

    def _load_data(self, data_dir):
        redshifts, masses = self._get_redshifts_and_masses(data_dir)
        thetas, kappas = self._get_thetas_and_kappas(data_dir)
        return thetas, kappas, redshifts, masses

    def _get_redshifts_and_masses(self, data_dir):
        zs, ms = np.loadtxt(data_dir + 'highSNR_z_mass.txt').T
        return zs, 1e14*ms

    def _get_thetas_and_kappas(self, data_dir):
        thetas, kappas = np.loadtxt(data_dir + 'highSNR_profile.txt').T
        thetas = thetas * self.FROM_ARCMIN
        return thetas, kappas
