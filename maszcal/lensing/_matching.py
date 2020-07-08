from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.nfw
import maszcal.concentration
import maszcal.matter
import maszcal.cosmo_utils
import maszcal.lensing._core as _core


@dataclass
class MatchingBaryonConvergenceModel(_core.MatchingBaryonModel):
    convergence_class: object = _core.MatchingGnfwBaryonConvergence
    sd_func: object = projector.sd

    def __post_init__(self):
        self._convergence = self.convergence_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.MatchingNfwModel,
            sd_func=self.sd_func,
        )
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _radius_space_kappa_total(self, rs, zs, mus, cons, alphas, betas, gammas, a_szs):
        return self._convergence.kappa_total(rs, zs, mus, cons, alphas, betas, gammas)

    def _angular_diameter_distance(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z).to(u.Mpc).value

    def kappa_total(self, thetas, cons, alphas, betas, gammas, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        radii_of_z = [thetas * self._angular_diameter_distance(z) for z in zs]
        kappas = np.array([
            self._radius_space_kappa_total(rs, zs[i:i+1], mus[i:i+1], cons, alphas, betas, gammas, a_szs)
            for i, rs in enumerate(radii_of_z)
        ]).squeeze()
        return kappas.reshape(thetas.shape + zs.shape + (-1,))

    def stacked_kappa(self, thetas, cons, alphas, betas, gammas, a_szs):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.kappa_total(thetas, cons, alphas, betas, gammas, a_szs).reshape(num_clusters, a_szs.size, thetas.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)


@dataclass
class MatchingBaryonShearModel(_core.MatchingBaryonModel):
    shear_class: object = _core.MatchingGnfwBaryonShear
    esd_func: object = projector.esd

    def __post_init__(self):
        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.MatchingNfwModel,
            esd_func=self.esd_func,
        )

    def delta_sigma_total(self, rs, cons, alphas, betas, gammas, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self._shear.delta_sigma_total(rs, zs, mus, cons, alphas, betas, gammas)

    def stacked_delta_sigma(self, rs, cons, alphas, betas, gammas, a_szs):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(rs, cons, alphas, betas, gammas, a_szs).reshape(num_clusters, a_szs.size, rs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)


@dataclass
class MatchingCmBaryonShearModel(MatchingBaryonShearModel):
    shear_class: object = _core.MatchingCmGnfwBaryonShear
    con_class: object = maszcal.concentration.ConModel
    esd_func: object = projector.esd

    def __post_init__(self):
        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.MatchingCmNfwModel,
            con_class=self.con_class,
            esd_func=self.esd_func,
        )

    def delta_sigma_total(self, rs, alphas, betas, gammas, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self._shear.delta_sigma_total(rs, zs, mus, alphas, betas, gammas)

    def stacked_delta_sigma(self, rs, alphas, betas, gammas, a_szs):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(rs, alphas, betas, gammas, a_szs).reshape(num_clusters, a_szs.size, rs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)
