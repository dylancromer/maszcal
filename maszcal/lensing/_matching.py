from dataclasses import dataclass
import numpy as np
import projector
import maszcal.nfw
import maszcal.concentration
import maszcal.matter
import maszcal.lensing._core as _core


@dataclass
class MatchingBaryonConvergenceModel(_core.BaryonModel):
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

    def kappa_total(self, rs, cons, alphas, betas, gammas, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self._convergence.kappa_total(rs, zs, mus, cons, alphas, betas, gammas)

    def stacked_kappa(self, rs, cons, alphas, betas, gammas, a_szs):
        'SHAPE a_sz, r, params'
        num_clusters = self.sz_masses.size
        profiles = self.kappa_total(rs, cons, alphas, betas, gammas, a_szs).reshape(num_clusters, a_szs.size, rs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[:, :, None, None] * profiles).sum(axis=0)


@dataclass
class MatchingBaryonShearModel(_core.BaryonModel):
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
