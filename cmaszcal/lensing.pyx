from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
from maszcal.cosmo_utils import get_astropy_cosmology
from maszcal.cosmology import CosmoParams
from maszcal.concentration import ConModel, MatchingConModel
import maszcal.nfw
import maszcal.mathutils


class MatchingGnfwBaryonShear:
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    MAX_INTEGRATION_RADIUS = 3.3
    NUM_INTEGRATION_RADII = 200

    def __init__(
            self,
            cosmo_params,
            mass_definition,
            delta,
            units,
            comoving_radii,
            nfw_class=maszcal.nfw.MatchingNfwModel,
    ):
        self.cosmo_params = cosmo_params
        self.mass_definition = mass_definition
        self.delta = delta
        self.units = units
        self.comoving_radii = comoving_radii
        self.nfw_class = nfw_class

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
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
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
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            projector.esd(rs, lambda r: self.rho_bary(r, zs, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )


@dataclass
class MatchingCmGnfwBaryonShear(MatchingGnfwBaryonShear):
    def __init__(
            self,
            cosmo_params,
            mass_definition,
            delta,
            units,
            comoving_radii,
            nfw_class=maszcal.nfw.MatchingCmNfwModel,
    ):
        self.cosmo_params = cosmo_params
        self.mass_definition = mass_definition
        self.delta = delta
        self.units = units
        self.comoving_radii = comoving_radii
        self.nfw_class = nfw_class

        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = MatchingConModel(mass_def, cosmology=self.cosmo_params)

    def _con(self, zs, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, zs, mass_def)

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
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

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
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            projector.esd(rs, lambda r: self.rho_bary(r, zs, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )

    def delta_sigma_total(self, rs, zs, mus, cons, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, zs, mus, cons, alphas, betas, gammas) + self.delta_sigma_cdm(rs, zs, mus, cons)


@dataclass
class MatchingBaryonShearModel:
    def __init__(
        self,
        sz_masses,
        redshifts,
        lensing_weights,
        cosmo_params=maszcal.cosmology.CosmoParams(),
        mass_definition='mean',
        delta=200,
        units=u.Msun/u.pc**2,
        comoving_radii=True,
        shear_class=MatchingGnfwBaryonShear,
    ):
        self.sz_masses = sz_masses
        self.redshifts = redshifts
        self.lensing_weights = lensing_weights
        self.cosmo_params = cosmo_params
        self.mass_definition = mass_definition
        self.delta = delta
        self.units = units
        self.comoving_radii = comoving_radii

        self._shear = shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
        )

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]

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
    def __init__(
        self,
        sz_masses,
        redshifts,
        lensing_weights,
        cosmo_params=maszcal.cosmology.CosmoParams(),
        mass_definition='mean',
        delta=200,
        units=u.Msun/u.pc**2,
        comoving_radii=True,
        shear_class=MatchingCmGnfwBaryonShear,
    ):
        self.sz_masses = sz_masses
        self.redshifts = redshifts
        self.lensing_weights = lensing_weights
        self.cosmo_params = cosmo_params
        self.mass_definition = mass_definition
        self.delta = delta
        self.units = units
        self.comoving_radii = comoving_radii

        self._shear = shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
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
