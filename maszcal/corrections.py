from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.cosmology
import maszcal.lensing
import maszcal.mathutils


@dataclass
class Matching2HaloBaryonShearModel:
    radii: np.ndarray
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    two_halo_term_function: object
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    shear_class: object = maszcal.lensing.MatchingGnfwBaryonShear

    def __post_init__(self):
        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
        )

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

    def delta_sigma_total(self, cons, alphas, betas, gammas, a_2hs, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        two_halo_delta_sigmas = self.delta_sigma_2_halo(zs, mus)
        one_halo_delta_sigmas = self._shear.delta_sigma_total(self.radii, zs, mus, cons, alphas, betas, gammas)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo_delta_sigmas, two_halo_delta_sigmas)

    def stacked_delta_sigma(self, cons, alphas, betas, gammas, a_2hs, a_szs):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(cons, alphas, betas, gammas, a_2hs, a_szs).reshape(num_clusters, a_szs.size, self.radii.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)
