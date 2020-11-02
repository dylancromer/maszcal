from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.concentration
import maszcal.matter
import maszcal.cosmo_utils
from . import _core


@dataclass
class MatchingConvergenceModel(_core.MatchingModel):
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    comoving: bool = True

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _radius_space_convergence(self, rs, zs, mus, *rho_params):
        return self.lensing_func(rs, zs, mus, *rho_params)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def _angular_diameter_distance(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z).to(u.Mpc).value

    def angle_scale_distance(self, z):
        if self.comoving:
            return self._comoving_distance(z)
        else:
            return self._angular_diameter_distance(z)

    def convergence(self, thetas, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        radii_of_z = thetas[:, None] * self.angle_scale_distance(zs)[None, :]
        convergences = self._radius_space_convergence(radii_of_z, zs, mus, *rho_params)
        return np.moveaxis(
            convergences,
            1,
            0,
        )

    def stacked_convergence(self, thetas, a_szs, *rho_params):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = np.moveaxis(
            self.convergence(thetas, a_szs, *rho_params),
            1,
            0,
        ).reshape(thetas.size, num_clusters, a_szs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[None, :, :, None] * profiles).sum(axis=1)


@dataclass
class ScatteredMatchingConvergenceModel(_core.ScatteredMatchingModel):
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    comoving: bool = True
    vectorized: bool = True

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)
        self.mus = np.log(np.geomspace(1e12, 6e15, self.num_mu_bins))

    def _radius_space_convergence(self, rs, zs, mus, *rho_params):
        return self.lensing_func(rs, zs, mus, *rho_params)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def _angular_diameter_distance(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z).to(u.Mpc).value

    def angle_scale_distance(self, z):
        if self.comoving:
            return self._comoving_distance(z)
        else:
            return self._angular_diameter_distance(z)

    def _get_mass_weights(self, mu_szs, a_szs):
        unnormalized_mass_weights = (self.prob_musz_given_mu(self.mus, mu_szs, a_szs)
                                     * self.logmass_prob_dist_func(self.redshifts, self.mus)[..., None])
        normalization = maszcal.mathutils.trapz_(unnormalized_mass_weights, axis=0, dx=np.gradient(self.mus))
        return unnormalized_mass_weights/normalization

    def _convergence_over_mass_range(self, thetas, *rho_params):
        radii_of_z = thetas[:, None] * self.angle_scale_distance(self.redshifts)[None, :]
        return self._radius_space_convergence(radii_of_z, self.redshifts, self.mus, *rho_params)

    def _convergence_over_mass_range_loop(self, thetas, mu, *rho_params):
        radii_of_z = thetas[:, None] * self.angle_scale_distance(self.redshifts)[None, :]
        return self._radius_space_convergence(radii_of_z, self.redshifts, mu, *rho_params).squeeze(axis=1)

    def _convergence_vectorized(self, thetas, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        convergences_over_mass_range = self._convergence_over_mass_range(thetas, *rho_params)
        return maszcal.mathutils.trapz_(
            convergences_over_mass_range[..., None, :]*mass_weights[None, ..., None],
            axis=1,
            dx=np.gradient(self.mus),
        )

    def _convergence_loop(self, thetas, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        convergence_test = self._convergence_over_mass_range_loop(thetas, self.mus[:1], *rho_params)
        convergences_over_mass_range = np.zeros(convergence_test.shape[:2] + a_szs.shape + convergence_test.shape[2:])
        dmus = np.gradient(self.mus)
        for i, mu in enumerate(self.mus):
            next_term = (self._convergence_over_mass_range_loop(thetas, np.array([mu]), *rho_params)[..., None, :]
                         * mass_weights[None, i, ..., None]) * dmus[i]

            if (i == 0) or (i == self.mus.size-1):
                next_term *= 1/2

            convergences_over_mass_range += next_term
        return convergences_over_mass_range

    def convergence(self, thetas, a_szs, *rho_params):
        if self.vectorized:
            return self._convergence_vectorized(thetas, a_szs, *rho_params)
        else:
            return self._convergence_loop(thetas, a_szs, *rho_params)

    def stacked_convergence(self, thetas, a_szs, *rho_params):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.convergence(thetas, a_szs, *rho_params)
        weights = self.normed_lensing_weights(a_szs)
        return (weights[None, :, None, None] * profiles).sum(axis=1)


class MatchingShearModel(_core.MatchingModel):

    def excess_surface_density_total(self, rs, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self.lensing_func(rs[:, None], zs, mus, *rho_params)

    def stacked_excess_surface_density(self, rs, a_szs, *rho_params):
        'SHAPE r, a_sz, params'
        num_clusters = self.sz_masses.size
        profiles = self.excess_surface_density_total(rs, a_szs, *rho_params).reshape(rs.size, num_clusters, a_szs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[None, :, :, None] * profiles).sum(axis=1)


@dataclass
class ScatteredMatchingShearModel(_core.ScatteredMatchingModel):
    vectorized: bool = True

    def __post_init__(self):
        self.mus = np.log(np.geomspace(1e12, 6e15, self.num_mu_bins))

    def _get_mass_weights(self, mu_szs, a_szs):
        unnormalized_mass_weights = (self.prob_musz_given_mu(self.mus, mu_szs, a_szs)
                                     * self.logmass_prob_dist_func(self.redshifts, self.mus)[..., None])
        normalization = maszcal.mathutils.trapz_(unnormalized_mass_weights, axis=0, dx=np.gradient(self.mus))
        return unnormalized_mass_weights/normalization

    def _excess_surface_density_total_vectorized(self, rs, a_szs, *rho_params):
        excess_surface_densities_over_mass_range = self.lensing_func(rs[:, None], self.redshifts, self.mus, *rho_params)
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        return maszcal.mathutils.trapz_(
            excess_surface_densities_over_mass_range[..., None, :]*mass_weights[None, ..., None],
            axis=1,
            dx=np.gradient(self.mus)
        )

    def _excess_surface_density_total_loop(self, rs, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        dmus = np.gradient(self.mus)

        def loop_func(mu): return self.lensing_func(rs[:, None], self.redshifts, mu, *rho_params).squeeze(axis=1)
        esd_test = loop_func(self.mus[:1])

        excess_surface_densities_over_mass_range = np.zeros(esd_test.shape[:2] + a_szs.shape + esd_test.shape[2:])
        for i, mu in enumerate(self.mus):
            next_term = (loop_func(np.array([mu]))[..., None, :]
                         * mass_weights[None, i, ..., None]) * dmus[i]

            if (i == 0) or (i == self.mus.size-1):
                next_term *= 1/2

            excess_surface_densities_over_mass_range += next_term
        return excess_surface_densities_over_mass_range

    def excess_surface_density_total(self, rs, a_szs, *rho_params):
        if self.vectorized:
            return self._excess_surface_density_total_vectorized(rs, a_szs, *rho_params)
        else:
            return self._excess_surface_density_total_loop(rs, a_szs, *rho_params)

    def stacked_excess_surface_density(self, rs, a_szs, *rho_params):
        'SHAPE r, a_sz, params'
        num_clusters = self.sz_masses.size
        profiles = self.excess_surface_density_total(rs, a_szs, *rho_params)
        weights = self.normed_lensing_weights(a_szs)
        return (weights[None, :, None, None] * profiles).sum(axis=1)
