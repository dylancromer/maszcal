from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.density
import maszcal.cosmology
import maszcal.lensing
import maszcal.mathutils
import maszcal.interp_utils


@dataclass
class SingleMass2HaloShearModel:
    radii: np.ndarray
    one_halo_rho_func: object
    one_halo_shear_class: object
    two_halo_term_function: object
    mass_definition: str = 'mean'
    delta: int = 200
    units: u.Quantity = u.Msun/u.pc**2
    esd_func: object = projector.esd

    def __post_init__(self):
        self._one_halo_shear = self.one_halo_shear_class(
            rho_func=self.one_halo_rho_func,
            units=self.units,
            esd_func=self.esd_func,
        )

    def _one_halo_excess_surface_density(self, zs, mus, *args):
        return self._one_halo_shear.excess_surface_density(self.radii, zs, mus, *args)

    def _two_halo_excess_surface_density(self, zs, mus):
        zs_twohalo, mus_twohalo = maszcal.interp_utils.cartesian_prod(zs, mus).T
        return np.swapaxes(
            self.two_halo_term_function(zs_twohalo, mus_twohalo).reshape(mus.size, zs.size, -1),
            -1,
            0,
        )

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo * a_2hs
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def excess_surface_density(self, a_2hs, zs, mus, *one_halo_params):
        one_halo = self._one_halo_excess_surface_density(zs, mus, *one_halo_params)
        two_halo = self._two_halo_excess_surface_density(zs, mus)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo, two_halo)


@dataclass
class Matching2HaloShearModel:
    radii: np.ndarray
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    one_halo_rho_func: object
    one_halo_shear_class: object
    two_halo_term_function: object
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    esd_func: object = projector.esd

    def __post_init__(self):
        self._one_halo_shear = self.one_halo_shear_class(
            rho_func=self.one_halo_rho_func,
            units=self.units,
            esd_func=self.esd_func,
        )

    def _one_halo_excess_surface_density(self, zs, mus, *args):
        return np.moveaxis(
            self._one_halo_shear.excess_surface_density(self.radii, zs, mus, *args),
            0,
            1,
        )

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]

    def _two_halo_excess_surface_density(self, zs, mus):
        return self.two_halo_term_function(zs, mus)

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo[..., None] * a_2hs[None, None, :]
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def excess_surface_density(self, a_2hs, a_szs, *one_halo_args):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        two_halo_excess_surface_densities = self._two_halo_excess_surface_density(zs, mus)
        one_halo_excess_surface_densities = self._one_halo_excess_surface_density(zs, mus, *one_halo_args)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo_excess_surface_densities, two_halo_excess_surface_densities)

    def stacked_excess_surface_density(self, a_2hs, a_szs, *one_halo_args):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.excess_surface_density(a_2hs, a_szs, *one_halo_args).reshape(num_clusters, a_szs.size, self.radii.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)


@dataclass
class Matching2HaloConvergenceModel:
    thetas: np.ndarray
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    one_halo_rho_func: object
    one_halo_convergence_class: object
    two_halo_term_function: object
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving: bool = True
    sd_func: object = projector.sd

    def __post_init__(self):
        self._one_halo_convergence_model = self.one_halo_convergence_class(
            rho_func=self.one_halo_rho_func,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving=self.comoving,
            sd_func=self.sd_func,
        )

    def _one_halo_convergence(self, zs, mus, *args):
        return np.moveaxis(
            self._one_halo_convergence_model.convergence(self.thetas, zs, mus, *args),
            0,
            1,
        )

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]

    def convergence_2_halo(self, zs, mus):
        return self.two_halo_term_function(zs, mus)

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo[..., None] * a_2hs[None, None, :]
        two_halo_indices = np.where(two_halo > one_halo)
        combination = one_halo.copy()
        combination[two_halo_indices] = two_halo[two_halo_indices]
        return combination

    def convergence(self, a_2hs, a_szs, *one_halo_args):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        two_halo_convergences = self.convergence_2_halo(zs, mus)
        one_halo_convergences = self._one_halo_convergence(zs, mus, *one_halo_args)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo_convergences, two_halo_convergences)

    def stacked_convergence(self, a_2hs, a_szs, *one_halo_args):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.convergence(a_2hs, a_szs, *one_halo_args).reshape(num_clusters, a_szs.size, self.thetas.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)
