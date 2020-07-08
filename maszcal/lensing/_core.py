from functools import partial
from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
import maszcal.nfw
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.concentration


@dataclass
class Gnfw:
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    MAX_INTEGRATION_RADIUS = 3.3
    NUM_INTEGRATION_RADII = 200

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object

    def __post_init__(self):
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter

    def mass_from_mu(self, mu):
        return np.exp(mu)

    def _init_nfw(self):
        self.nfw_model = self.nfw_class(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def _r_delta(self, zs, mus):
        '''
        SHAPE mu, z
        '''
        masses = self.mass_from_mu(mus)
        try:
            return self.nfw_model.radius_delta(zs, masses)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.radius_delta(zs, masses)

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, rs.shape, params
        '''
        ys = (rs[None, None, ...]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+2)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = alphas.reshape((rs.ndim + 2)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 2)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 2)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[None, None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[None, None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _move_radius_axes_to_front(self, arr, start, stop):
        radius_axes = np.arange(arr.ndim)[start:stop]
        new_radius_axes = np.arange(radius_axes.size)
        return np.moveaxis(
            arr,
            tuple(radius_axes),
            tuple(new_radius_axes),
        )

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        profile_shape = self._move_radius_axes_to_front(profile_shape, 2, -1)
        return norm * profile_shape

    def _rho_nfw(self, rs, zs, mus, cons):
        masses = self.mass_from_mu(mus)

        try:
            return self.nfw_model.rho(rs, zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.rho(rs, zs, masses, cons)

    def rho_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, zs, mus, cons, alphas, betas, gammas)

    def rho_cdm(self, rs, zs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, zs, mus, cons)


@dataclass
class GnfwBaryonShear(Gnfw):
    esd_func: object

    def delta_sigma_cdm(self, rs, zs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)

        try:
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, zs, masses, cons)

    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.rho_bary(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, zs, mus, cons, alphas, betas, gammas) + self.delta_sigma_cdm(rs, zs, mus, cons)


@dataclass
class SingleMassGnfwBaryonShear(GnfwBaryonShear):
    nfw_class: object

    def _r_delta(self, zs, mus):
        '''
        SHAPE z, params
        '''
        masses = self.mass_from_mu(mus)
        try:
            return self.nfw_model.radius_delta(zs, masses).T
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.radius_delta(zs, masses).T

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE rs.shape, z, params
        '''
        ys = (rs[..., None, None]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+2, append_dims=False)) / self.CORE_RADIUS

        alphas = alphas.reshape((rs.ndim + 1)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 1)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 1)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[None, ..., None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)[None, ...]
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, r, params
        '''
        return (self.esd_func(rs, lambda r: self.rho_bary(r, zs, mus, cons, alphas, betas, gammas))
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
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    MAX_INTEGRATION_RADIUS = 3.3
    NUM_INTEGRATION_RADII = 200

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    con_class: object
    nfw_class: object

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def _con(self, zs, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, zs, mass_def)

    def _gnfw_norm(self, zs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, zs, mus)[..., None] * rs[None, None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[None, None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, zs, mus, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        profile_shape = self._move_radius_axes_to_front(profile_shape, 2, -1)
        return norm * profile_shape

    def _rho_nfw(self, rs, zs, mus):
        masses = self.mass_from_mu(mus)
        cons = self._con(zs, masses)

        try:
            return self.nfw_model.rho(rs, zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.rho(rs, zs, masses, cons)

    def rho_bary(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, zs, mus, alphas, betas, gammas)

    def rho_cdm(self, rs, zs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, zs, mus)

    def delta_sigma_cdm(self, rs, zs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)
        cons = self._con(zs, masses)

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
            self.esd_func(rs, lambda r: self.rho_bary(r, zs, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, zs, mus, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, zs, mus, alphas, betas, gammas) + self.delta_sigma_cdm(rs, zs, mus)[..., None]


@dataclass
class MatchingGnfwBaryonConvergence(Gnfw):
    CMB_REDSHIFT = 1100

    nfw_class: object
    sd_func: object

    def __post_init__(self):
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, rs.shape, params
        '''
        ys = (rs[None, ...]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+1)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = alphas.reshape((rs.ndim + 1)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 1)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 1)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)

        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        profile_shape = self._move_radius_axes_to_front(profile_shape, 1, -1)
        return norm * profile_shape

    def _rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self._move_radius_axes_to_front(self.rho_cdm(rs, zs, mus, cons), 1, -1)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        try:
            return self._rho_tot(rs, zs, mus, cons, alphas, betas, gammas)
        except AttributeError:
            self._init_nfw()
            return self._rho_tot(rs, zs, mus, cons, alphas, betas, gammas)

    def kappa_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        return np.moveaxis(
            self.sd_func(rs, lambda r: self.rho_tot(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        ) / self.sigma_crit(z_lens=zs)[:, None, None]


@dataclass
class MatchingGnfwBaryonShear(GnfwBaryonShear):
    nfw_class: object

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, rs.shape, params
        '''
        ys = (rs[None, ...]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+1)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = alphas.reshape((rs.ndim + 1)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 1)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 1)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)

        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        profile_shape = self._move_radius_axes_to_front(profile_shape, 1, -1)
        return norm * profile_shape

    def delta_sigma_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.rho_bary(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )


@dataclass
class MatchingCmGnfwBaryonShear(CmGnfwBaryonShear):
    nfw_class: object

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, rs.shape, params
        '''
        ys = (rs[None, ...]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+1)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = alphas.reshape((rs.ndim + 1)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 1)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 1)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)

        top_integrand = self._rho_nfw(rs, zs, mus)[..., None] * rs[None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, zs, mus, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)

        radius_axes = np.arange(profile_shape.ndim)[1:-1]
        new_radius_axes = np.arange(radius_axes.size)
        profile_shape = np.moveaxis(
            profile_shape,
            tuple(radius_axes),
            tuple(new_radius_axes),
        )

        return norm * profile_shape

    def delta_sigma_bary(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            self.esd_func(rs, lambda r: self.rho_bary(r, zs, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
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
