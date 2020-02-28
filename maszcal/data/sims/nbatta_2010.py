import numpy as np
import maszcal.cosmology
from maszcal.data.templates import WeakLensingData


class NBatta2010(WeakLensingData):
    def __init__(self, data_dir='data/NBatta2010/', covariances=None):
        self.cosmology = self._init_cosmology()

        rs, wl_signals, zs = self._load_data(data_dir)

        if covariances is None:
            covariance = 0.01 * np.identity(rs.size)
            num_clusters = wl_signals.shape[2]
            covariances = np.stack(num_clusters*(covariance,), axis=-1)

        super().__init__(
            radii=rs,
            redshifts=zs,
            wl_signals=wl_signals,
            covariances=covariances,
        )

    def _init_cosmology(self):
        omega_bary = 0.044
        omega_matter = 0.25
        omega_cdm = omega_matter - omega_bary
        h=0.72
        return maszcal.cosmology.CosmoParams(
            hubble_constant=100*h,
            omega_bary_hsqr=omega_bary * h**2,
            omega_cdm_hsqr=omega_cdm * h**2,
            spectral_index=0.96,
            sigma_8=0.8,
            omega_bary=omega_bary,
            omega_cdm=omega_cdm,
            omega_matter=omega_matter,
            omega_lambda=1-omega_matter,
            h=h,
        )

    def __load_cluster_redshift(self, data_dir, snapshot_number):
        cluster_as = np.loadtxt(data_dir + 'outputs_selection.165.txt')[::-1]
        zs = 1./cluster_as[6:] - 1.
        snap_redshift = zs[snapshot_number]
        return snap_redshift

    def __load_wl_signal_and_radii(self, data_dir, snapshot_number, num_clusters, num_radii):
        file_2 = data_dir + 'DS_data/raw_' + str(snapshot_number) + '_comovLRG.dat'
        with open(file_2, 'rb') as file_:
            temp = np.fromfile(file=file_, dtype=np.float32)

        wl_array = np.reshape(temp, (num_radii, 2, num_clusters))

        rs = wl_array[:, 0, :] / self.cosmology.h
        ### Mpc / h
        wl_esds = wl_array[:, 1, :] * 1e4 * self.cosmology.h
        ### h M_sun / pc
        return rs, wl_esds

    def _reduce_radii(self, radii):
        if np.all(radii[:, 0] == radii.T):
            return radii[:, 0]
        else:
            raise ValueError('radii are different for different clusters')

    def _load_data(self, data_dir):
        NUM_SNAPS = 55 - 41
        NUM_CLUSTERS = 100
        NUM_RADII = 40

        zs = np.zeros(NUM_SNAPS)
        wl_esds = np.zeros((NUM_RADII, NUM_CLUSTERS, NUM_SNAPS))

        for i, snap in enumerate(range(41, 55)):
            zs[i] = self.__load_cluster_redshift(data_dir, snap)
            rs, wl_esds[..., i] = self.__load_wl_signal_and_radii(data_dir, snap, NUM_CLUSTERS, NUM_RADII)

        wl_esds = np.moveaxis(wl_esds, 2, 1)
        rs = self._reduce_radii(rs)

        return rs, wl_esds, zs
