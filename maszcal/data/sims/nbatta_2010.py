import numpy as np
import maszcal.cosmology
from maszcal.data.templates import WeakLensingData


class NBatta2010(WeakLensingData):
    def __init__(self, data_dir='data/NBatta2010/'):
        self.cosmology = self._init_cosmology()

        rs, wl_signals, zs, masses = self._load_data(data_dir)

        super().__init__(
            radii=rs,
            redshifts=zs,
            wl_signals=wl_signals,
            masses=masses,
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

    @staticmethod
    def from_data(radii, redshifts, wl_signals, masses, cosmology):
        wl_data_instance = WeakLensingData(
            radii=radii,
            redshifts=redshifts,
            wl_signals=wl_signals,
            masses=masses,
        )

        wl_data_instance.cosmology = cosmology

        return wl_data_instance

    def __load_cluster_masses(self, data_dir, snapshot_number, num_clusters):
        file_1 = data_dir + 'cluster_info/CLUSTER_MAP_INFO3PBC.L165.256.FBN2_' + str(snapshot_number) + '500.d'
        with open(file_1, 'rb') as file_:
            temp = np.fromfile(file=file_, dtype=np.float32)
        clust_data = np.reshape(temp, (num_clusters, 4))
        #Simulation number, cluster id in simulation, Mass M500 in 1e10 Msol / h, R500 in kpc/h
        true_masses = clust_data[:, 2] * 1e10 / self.cosmology.h
        return true_masses

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

    def cut_radii(self, lower_radius, upper_radius):
        cut_indices = np.where((lower_radius <= self.radii) & (self.radii <= upper_radius))
        return self.from_data(
            radii=self.radii[cut_indices],
            redshifts=self.redshifts,
            wl_signals=self.wl_signals[cut_indices],
            masses=self.masses,
            cosmology=self.cosmology,
        )

    def _load_data(self, data_dir):
        NUM_SNAPS = 55 - 41
        NUM_CLUSTERS = 100
        NUM_RADII = 40

        zs = np.zeros(NUM_SNAPS)
        wl_esds = np.zeros((NUM_RADII, NUM_CLUSTERS, NUM_SNAPS))
        masses = np.zeros((NUM_CLUSTERS, NUM_SNAPS))

        for i, snap in enumerate(range(41, 55)):
            zs[i] = self.__load_cluster_redshift(data_dir, snap)
            rs, wl_esds[..., i] = self.__load_wl_signal_and_radii(data_dir, snap, NUM_CLUSTERS, NUM_RADII)
            masses[..., i] = self.__load_cluster_masses(data_dir, snap, NUM_CLUSTERS)

        wl_esds = np.moveaxis(wl_esds, 2, 1)
        masses = masses.T
        rs = self._reduce_radii(rs)

        return rs, wl_esds, zs, masses
