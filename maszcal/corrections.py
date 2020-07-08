from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.nfw
import maszcal.cosmology
import maszcal.lensing
import maszcal.mathutils


@dataclass
class Matching2HaloShearModel:
    radii: np.ndarray
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    one_halo_shear_class: object
    two_halo_term_function: object
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    esd_func: object = projector.esd

    def __post_init__(self):
        self._one_halo_shear = self.one_halo_shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.MatchingNfwModel,
            esd_func=self.esd_func,
        )

    def _one_halo_delta_sigma(self, zs, mus, *args):
        return self._one_halo_shear.delta_sigma_total(self.radii, zs, mus, *args)

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]

    def delta_sigma_2_halo(self, zs, mus):
        return self.two_halo_term_function(zs, mus)

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo[..., None] * a_2hs[None, None, :]
        two_halo_indices = np.where(two_halo > one_halo)
        combination = one_halo.copy()
        combination[two_halo_indices] = two_halo[two_halo_indices]
        return combination

    def delta_sigma_total(self, a_2hs, a_szs, *one_halo_args):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        two_halo_delta_sigmas = self.delta_sigma_2_halo(zs, mus)
        one_halo_delta_sigmas = self._one_halo_delta_sigma(zs, mus, *one_halo_args)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo_delta_sigmas, two_halo_delta_sigmas)

    def stacked_delta_sigma(self, a_2hs, a_szs, *one_halo_args):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(a_2hs, a_szs, *one_halo_args).reshape(num_clusters, a_szs.size, self.radii.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)
