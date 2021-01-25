import numpy as np
import maszcal.cosmology
from maszcal.data.templates import StackedWeakLensingData


class ActTestData(StackedWeakLensingData):
    def __init__(self, data_dir='data/test-act/'):
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
        thetas = self._get_thetas(data_dir)
        kappas = self._get_kappas(data_dir)
        return thetas, kappas, redshifts, masses

    def _get_redshifts_and_masses(self, data_dir):
        bin_0 = np.loadtxt(data_dir + '0ycbin050.txt')
        bin_1 = np.loadtxt(data_dir + '1ycbin050075.txt')
        bin_2 = np.loadtxt(data_dir + '2ycbin075100.txt')
        bin_3 = np.loadtxt(data_dir + '3ycbin100.txt')
        #zs, ms = np.concatenate((bin_0, bin_1, bin_2, bin_3), axis=0).T
        zs, ms = bin_2.T
        return zs, 1e14*ms

    def _get_thetas(self, data_dir):
        return np.load(data_dir + 'test-act_thetas.npy')

    def _get_kappas(self, data_dir):
        return np.load(data_dir + 'test-act_filtered-kappas.npy')
