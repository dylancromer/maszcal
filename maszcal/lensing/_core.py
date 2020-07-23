from functools import partial
from dataclasses import dataclass
import numpy as np
from astropy import units as u
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.concentration


@dataclass
class Shear:
    rho_func: object
    units: u.Quantity
    esd_func: object

    def delta_sigma_total(self, rs, zs, mus, *rho_params):
        return self.esd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
        ) * (u.Msun/u.Mpc**2).to(self.units)


@dataclass
class MatchingConvergence:
    CMB_REDSHIFT = 1100

    rho_func: object
    cosmo_params: maszcal.cosmology.CosmoParams
    units: u.Quantity
    sd_func: object

    def __post_init__(self):
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def kappa_total(self, rs, zs, mus, *rho_params):
        return self.sd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
        ) * (u.Msun/u.Mpc**2).to(self.units) / self.sigma_crit(z_lens=zs)[None, :, None]


@dataclass
class MatchingModel:
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    rho_func: object
    units: u.Quantity = u.Msun/u.pc**2

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]
