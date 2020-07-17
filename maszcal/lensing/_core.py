from functools import partial
from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
import maszcal.nfw
import maszcal.gnfw
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.concentration


@dataclass
class GnfwBaryonShear:
    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object
    gnfw_class: object
    esd_func: object

    def _init_nfw(self):
        self.nfw_model = self.nfw_class(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def _init_gnfw(self):
        self.gnfw = self.gnfw_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_model=self.nfw_model,
        )

    def __post_init__(self):
        self._init_nfw()
        self._init_gnfw()
        self.baryon_frac = self.gnfw.baryon_frac

    def mass_from_mu(self, mu):
        return self.gnfw.mass_from_mu(mu)

    def delta_sigma_cdm(self, rs, zs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)
        return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, zs, masses, cons)

    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.gnfw.rho_bary(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, zs, mus, cons, alphas, betas, gammas) + self.delta_sigma_cdm(rs, zs, mus, cons)


class SingleMassGnfwBaryonShear(GnfwBaryonShear):
    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, r, params
        '''
        return (self.esd_func(rs, lambda r: self.gnfw.rho_bary(r, zs, mus, cons, alphas, betas, gammas))
                * (u.Msun/u.Mpc**2).to(self.units))

    def delta_sigma_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        delta_sigma_bary = np.moveaxis(
            self.delta_sigma_bary(rs, zs, mus, cons, alphas, betas, gammas),
            -2,
            0,
        )
        return delta_sigma_bary + self.delta_sigma_cdm(rs, zs, mus, cons)


@dataclass
class CmGnfwBaryonShear(GnfwBaryonShear):
    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    con_class: object
    nfw_class: object
    gnfw_class: object
    esd_func: object

    def _init_gnfw(self):
        self.gnfw = self.gnfw_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            con_class=self.con_class,
            nfw_model=self.nfw_model,
        )

    def delta_sigma_cdm(self, rs, zs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)
        cons = self.gnfw._con(zs, masses)

        try:
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, zs, masses, cons)

    def delta_sigma_bary(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.gnfw.rho_bary(r, zs, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, zs, mus, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, zs, mus, alphas, betas, gammas) + self.delta_sigma_cdm(rs, zs, mus)[..., None]


@dataclass
class MatchingGnfwBaryonConvergence:
    CMB_REDSHIFT = 1100

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object
    gnfw_class: object
    sd_func: object

    def _init_nfw(self):
        self.nfw_model = self.nfw_class(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def _init_gnfw(self):
        self.gnfw = self.gnfw_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_model=self.nfw_model,
        )

    def __post_init__(self):
        self._init_nfw()
        self._init_gnfw()
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def kappa_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        return np.moveaxis(
            self.sd_func(rs, lambda r: self.gnfw.rho_tot(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        ) / self.sigma_crit(z_lens=zs)[:, None, None]


@dataclass
class MiscenteredMatchingGnfwBaryonConvergence(MatchingGnfwBaryonConvergence):
    CMB_REDSHIFT = 1100

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object
    gnfw_class: object
    sd_func: object
    miscentering_func: object

    def _init_gnfw(self):
        self.gnfw = self.gnfw_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_model=self.nfw_model,
            miscentering_func=self.miscentering_func,
        )

    def kappa_total(self, rs, zs, mus, cons, alphas, betas, gammas, misc_scales):
        return np.moveaxis(
            self.sd_func(rs, lambda r: self.gnfw.rho_tot(r, zs, mus, cons, alphas, betas, gammas, misc_scales)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        ) / self.sigma_crit(z_lens=zs)[:, None, None]


class MatchingGnfwBaryonShear(GnfwBaryonShear):
    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.gnfw.rho_bary(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )


@dataclass
class MatchingCmGnfwBaryonShear(CmGnfwBaryonShear):
    nfw_class: object

    def delta_sigma_bary(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.gnfw.rho_bary(r, zs, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )


@dataclass
class MatchingBaryonModel:
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]
