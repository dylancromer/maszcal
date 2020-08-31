from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.concentration
import maszcal.matter
import maszcal.cosmo_utils
import maszcal.lensing._core as _core


@dataclass
class MatchingConvergenceModel(_core.MatchingModel):
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    comoving: bool = True

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _radius_space_kappa(self, rs, zs, mus, *rho_params):
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

    def kappa(self, thetas, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        radii_of_z = [thetas * self.angle_scale_distance(z) for z in zs]
        kappas = np.array([
            self._radius_space_kappa(rs, zs[i:i+1], mus[i:i+1], *rho_params)
            for i, rs in enumerate(radii_of_z)
        ]).squeeze(axis=2)
        return np.moveaxis(
            kappas,
            1,
            0,
        )

    def stacked_kappa(self, thetas, a_szs, *rho_params):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.kappa(thetas, a_szs, *rho_params).reshape(thetas.size, num_clusters, a_szs.size, -1)
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

    def _radius_space_kappa(self, rs, zs, mus, *rho_params):
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

    def _kappa_over_mass_range(self, thetas, *rho_params):
        radii_of_z = [thetas * self.angle_scale_distance(z) for z in self.redshifts]
        kappas = np.array([
            self._radius_space_kappa(rs, self.redshifts[i:i+1], self.mus, *rho_params)
            for i, rs in enumerate(radii_of_z)
        ]).squeeze(axis=3)
        return np.moveaxis(
            kappas,
            (1, 2),
            (0, 1),
        )

    def _kappa_over_mass_range_loop(self, thetas, mu, *rho_params):
        radii_of_z = [thetas * self.angle_scale_distance(z) for z in self.redshifts]
        kappas = np.array([
            self._radius_space_kappa(rs, self.redshifts[i:i+1], mu, *rho_params)
            for i, rs in enumerate(radii_of_z)
        ]).squeeze(axis=(2, 3))
        return np.moveaxis(
            kappas,
            1,
            0,
        )

    def _kappa_vectorized(self, thetas, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        kappas_over_mass_range = self._kappa_over_mass_range(thetas, *rho_params)
        return maszcal.mathutils.trapz_(
            kappas_over_mass_range[..., None, :]*mass_weights[None, ..., None],
            axis=1,
            dx=np.gradient(self.mus),
        )

    def _kappa_loop(self, thetas, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        kappa_test = self._kappa_over_mass_range_loop(thetas, self.mus[:1], *rho_params)
        kappas_over_mass_range = np.zeros(kappa_test.shape[:2] + a_szs.shape + kappa_test.shape[2:])
        dmus = np.gradient(self.mus)
        for i, mu in enumerate(self.mus):
            next_term = (self._kappa_over_mass_range_loop(thetas, np.array([mu]), *rho_params)[..., None, :]
                         * mass_weights[None, i, ..., None]) * dmus[i]

            if (i == 0) or (i == self.mus.size-1):
                next_term *= 1/2

            kappas_over_mass_range += next_term
        return kappas_over_mass_range

    def kappa(self, thetas, a_szs, *rho_params):
        if self.vectorized:
            return self._kappa_vectorized(thetas, a_szs, *rho_params)
        else:
            return self._kappa_loop(thetas, a_szs, *rho_params)

    def stacked_kappa(self, thetas, a_szs, *rho_params):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.kappa(thetas, a_szs, *rho_params)
        weights = self.normed_lensing_weights(a_szs)
        return (weights[None, :, None, None] * profiles).sum(axis=1)


class MatchingShearModel(_core.MatchingModel):

    def delta_sigma_total(self, rs, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self.lensing_func(rs, zs, mus, *rho_params)

    def stacked_delta_sigma(self, rs, a_szs, *rho_params):
        'SHAPE r, a_sz, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(rs, a_szs, *rho_params).reshape(rs.size, num_clusters, a_szs.size, -1)
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

    def _delta_sigma_total_vectorized(self, rs, a_szs, *rho_params):
        delta_sigmas_over_mass_range = self.lensing_func(rs, self.redshifts, self.mus, *rho_params)
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        return maszcal.mathutils.trapz_(
            delta_sigmas_over_mass_range[..., None, :]*mass_weights[None, ..., None],
            axis=1,
            dx=np.gradient(self.mus)
        )

    def _delta_sigma_total_loop(self, rs, a_szs, *rho_params):
        mu_szs = np.log(self.sz_masses)
        mass_weights = self._get_mass_weights(mu_szs, a_szs)
        dmus = np.gradient(self.mus)

        def loop_func(mu): return self.lensing_func(rs, self.redshifts, mu, *rho_params).squeeze(axis=1)
        esd_test = loop_func(self.mus[:1])

        delta_sigmas_over_mass_range = np.zeros(esd_test.shape[:2] + a_szs.shape + esd_test.shape[2:])
        for i, mu in enumerate(self.mus):
            next_term = (loop_func(np.array([mu]))[..., None, :]
                         * mass_weights[None, i, ..., None]) * dmus[i]

            if (i == 0) or (i == self.mus.size-1):
                next_term *= 1/2

            delta_sigmas_over_mass_range += next_term
        return delta_sigmas_over_mass_range

    def delta_sigma_total(self, rs, a_szs, *rho_params):
        if self.vectorized:
            return self._delta_sigma_total_vectorized(rs, a_szs, *rho_params)
        else:
            return self._delta_sigma_total_loop(rs, a_szs, *rho_params)

    def stacked_delta_sigma(self, rs, a_szs, *rho_params):
        'SHAPE r, a_sz, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(rs, a_szs, *rho_params)
        weights = self.normed_lensing_weights(a_szs)
        return (weights[None, :, None, None] * profiles).sum(axis=1)
