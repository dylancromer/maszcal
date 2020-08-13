from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import meso
import maszcal.concentration
import maszcal.matter
import maszcal.cosmo_utils
import maszcal.lensing._core as _core


@dataclass
class MatchingConvergenceModel(_core.MatchingModel):
    convergence_class: object = _core.Convergence
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    sd_func: object = projector.sd

    def __post_init__(self):
        self._convergence = self.convergence_class(
            rho_func=self.rho_func,
            cosmo_params=self.cosmo_params,
            units=self.units,
            sd_func=self.sd_func,
        )
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _radius_space_kappa(self, rs, zs, mus, *rho_params):
        return self._convergence.kappa(rs, zs, mus, *rho_params)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def kappa(self, thetas, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        radii_of_z = [thetas * self._comoving_distance(z) for z in zs]
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
class MatchingShearModel(_core.MatchingModel):
    shear_class: object = _core.Shear
    esd_func: object = projector.esd

    def __post_init__(self):
        self._shear = self.shear_class(
            rho_func=self.rho_func,
            units=self.units,
            esd_func=self.esd_func,
        )

    def delta_sigma_total(self, rs, a_szs, *rho_params):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        zs = np.repeat(self.redshifts, a_szs.size)
        return self._shear.delta_sigma_total(rs, zs, mus, *rho_params)

    def stacked_delta_sigma(self, rs, a_szs, *rho_params):
        'SHAPE r, a_sz, params'
        num_clusters = self.sz_masses.size
        profiles = self.delta_sigma_total(rs, a_szs, *rho_params).reshape(rs.size, num_clusters, a_szs.size, -1)
        weights = self.normed_lensing_weights(a_szs).reshape(num_clusters, a_szs.size)
        return (weights[None, :, :, None] * profiles).sum(axis=1)
