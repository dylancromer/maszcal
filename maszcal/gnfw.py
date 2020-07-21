from functools import partial
from dataclasses import dataclass
import numpy as np
import astropy.units as u
import meso
import maszcal.cosmology
import maszcal.mathutils


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
    nfw_model: object

    def __post_init__(self):
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter

    def mass_from_mu(self, mu):
        return np.exp(mu)

    def _r_delta(self, zs, mus):
        '''
        SHAPE mu, z
        '''
        masses = self.mass_from_mu(mus)
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

    def _rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self._move_radius_axes_to_front(self.rho_cdm(rs, zs, mus, cons), 1, -1)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm


class SingleMassGnfw(Gnfw):
    def _r_delta(self, zs, mus):
        '''
        SHAPE z, params
        '''
        masses = self.mass_from_mu(mus)
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


@dataclass
class CmGnfw(Gnfw):
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
    nfw_model: object

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


@dataclass
class MatchingGnfw(Gnfw):
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


@dataclass
class MatchingMiscenteredGnfw:
    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_model: object
    miscentering_func: object = meso.Rho().miscenter

    def _init_gnfw(self):
        self.gnfw = MatchingGnfw(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_model=self.nfw_model,
        )

    def __post_init__(self):
        self._init_gnfw()

    def _move_radius_axes_to_front(self, arr, start, stop):
        return self.gnfw._move_radius_axes_to_front(arr, start, stop)

    def _rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self._move_radius_axes_to_front(self.gnfw.rho_cdm(rs, zs, mus, cons), 1, -1)
        return self.gnfw.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm

    def _miscentering_dist(self, rs, misc_scales):
        misc_scales = maszcal.mathutils.atleast_kd(misc_scales, rs.ndim+1, append_dims=False)
        rs = rs[..., None]
        return (rs/misc_scales**2) * np.exp(-(rs/misc_scales)**2 / 2)

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas, misc_scales):
        '''
        SHAPE mu, z, r, params
        '''
        return self.miscentering_func(
            rs,
            lambda r: self._rho_tot(r, zs, mus, cons, alphas, betas, gammas),
            lambda r: self._miscentering_dist(r, misc_scales),
        )


class MatchingConvergenceGnfw(MatchingGnfw):
    def _rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self._move_radius_axes_to_front(self.rho_cdm(rs, zs, mus, cons), 1, -1)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return self._rho_tot(rs, zs, mus, cons, alphas, betas, gammas)


class MatchingCmGnfw(CmGnfw):
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
