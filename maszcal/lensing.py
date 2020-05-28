from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
from maszcal.tinker import TinkerHmf
from maszcal.cosmo_utils import get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.concentration import ConModel
import maszcal.nfw
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults


@dataclass
class GnfwBaryonShear:
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    MAX_INTEGRATION_RADIUS = 3.3
    NUM_INTEGRATION_RADII = 200

    zs: np.ndarray
    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object = maszcal.nfw.NfwModel

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

    def _r_delta(self, mus):
        '''
        SHAPE mu, z
        '''
        masses = self.mass_from_mu(mus)
        try:
            return self.nfw_model.radius_delta(self.zs, masses)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.radius_delta(self.zs, masses)

    def gnfw_shape(self, rs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, rs.shape, params
        '''
        ys = (rs[None, None, ...]/maszcal.mathutils.atleast_kd(self._r_delta(mus), rs.ndim+2)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = alphas.reshape((rs.ndim + 2)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 2)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 2)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, mus, cons) * rs[None, None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, mus, alphas, betas, gammas) * rs[None, None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(mus, cons, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, mus, alphas, betas, gammas)

        radius_axes = np.arange(profile_shape.ndim)[2:-1]
        new_radius_axes = np.arange(radius_axes.size)
        profile_shape = np.moveaxis(
            profile_shape,
            tuple(radius_axes),
            tuple(new_radius_axes),
        )

        return norm * profile_shape

    def _rho_nfw(self, rs, mus, cons):
        masses = self.mass_from_mu(mus)

        try:
            return self.nfw_model.rho(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.rho(rs, self.zs, masses, cons)

    def rho_bary(self, rs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, mus, cons, alphas, betas, gammas)

    def rho_cdm(self, rs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, mus, cons)

    def delta_sigma_cdm(self, rs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)

        try:
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, self.zs, masses, cons)

    def delta_sigma_bary(self, rs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return np.moveaxis(
            projector.esd(rs, lambda r: self.rho_bary(r, mus, cons, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, mus, cons, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, mus, cons, alphas, betas, gammas) + self.delta_sigma_cdm(rs, mus, cons)


@dataclass
class SingleMassGnfwBaryonShear(GnfwBaryonShear):
    nfw_class: object = maszcal.nfw.SingleMassNfwModel

    def _r_delta(self, mus):
        '''
        SHAPE z, params
        '''
        masses = self.mass_from_mu(mus)
        try:
            return self.nfw_model.radius_delta(self.zs, masses).T
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.radius_delta(self.zs, masses).T

    def gnfw_shape(self, rs, mus, alphas, betas, gammas):
        '''
        SHAPE rs.shape, z, params
        '''
        ys = (rs[..., None, None]/maszcal.mathutils.atleast_kd(self._r_delta(mus), rs.ndim+2, append_dims=False)) / self.CORE_RADIUS

        alphas = alphas.reshape((rs.ndim + 1)*(1,) + (alphas.size,))
        betas = betas.reshape((rs.ndim + 1)*(1,) + (betas.size,))
        gammas = gammas.reshape((rs.ndim + 1)*(1,) + (gammas.size,))

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, mus, cons) * rs[None, ..., None]**2
        bottom_integrand = self.gnfw_shape(rs, mus, alphas, betas, gammas) * rs[..., None, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(mus, cons, alphas, betas, gammas)[None, ...]
        profile_shape = self.gnfw_shape(rs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def delta_sigma_bary(self, rs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, r, params
        '''
        return (projector.esd(rs, lambda r: self.rho_bary(r, mus, cons, alphas, betas, gammas))
                * (u.Msun/u.Mpc**2).to(self.units))

    def delta_sigma_total(self, rs, mus, cons, alphas, betas, gammas):
        delta_sigma_bary = np.moveaxis(
            self.delta_sigma_bary(rs, mus, cons, alphas, betas, gammas),
            -2,
            0,
        )
        return delta_sigma_bary + self.delta_sigma_cdm(rs, mus, cons)


@dataclass
class CmGnfwBaryonShear(GnfwBaryonShear):
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    MAX_INTEGRATION_RADIUS = 3.3
    NUM_INTEGRATION_RADII = 200

    zs: np.ndarray
    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    units: u.Quantity
    comoving_radii: bool
    nfw_class: object = maszcal.nfw.NfwCmModel

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = ConModel(mass_def, cosmology=self.cosmo_params)

    def _con(self, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, self.zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, self.zs, mass_def)

    def _gnfw_norm(self, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, mus)[..., None] * rs[None, None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, mus, alphas, betas, gammas) * rs[None, None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def _rho_gnfw(self, rs, mus, alphas, betas, gammas):
        norm = self._gnfw_norm(mus, alphas, betas, gammas)
        norm = norm.reshape(rs.ndim*(1,) + norm.shape)
        profile_shape = self.gnfw_shape(rs, mus, alphas, betas, gammas)

        radius_axes = np.arange(profile_shape.ndim)[2:-1]
        new_radius_axes = np.arange(radius_axes.size)
        profile_shape = np.moveaxis(
            profile_shape,
            tuple(radius_axes),
            tuple(new_radius_axes),
        )

        return norm * profile_shape

    def _rho_nfw(self, rs, mus):
        masses = self.mass_from_mu(mus)
        cons = self._con(masses)

        try:
            return self.nfw_model.rho(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.rho(rs, self.zs, masses, cons)

    def rho_bary(self, rs, mus, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, mus, alphas, betas, gammas)

    def rho_cdm(self, rs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, mus)

    def delta_sigma_cdm(self, rs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass_from_mu(mus)
        cons = self._con(masses)

        try:
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return (1-self.baryon_frac) * self.nfw_model.delta_sigma(rs, self.zs, masses, cons)

    def delta_sigma_bary(self, rs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, r, params
        '''
        return np.moveaxis(
            projector.esd(rs, lambda r: self.rho_bary(r, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            2,
        )

    def delta_sigma_total(self, rs, mus, alphas, betas, gammas):
        return self.delta_sigma_bary(rs, mus, alphas, betas, gammas) + self.delta_sigma_cdm(rs, mus)[..., None]


@dataclass
class MatchingCmGnfwBaryonShear(CmGnfwBaryonShear):
    nfw_class: object = maszcal.nfw.MatchingNfwModel

    def _gnfw_norm(self, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rs = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.MAX_INTEGRATION_RADIUS,
            self.NUM_INTEGRATION_RADII,
        )

        drs = np.gradient(rs)
        top_integrand = self._rho_nfw(rs, mus)[..., None] * rs[None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, mus, alphas, betas, gammas) * rs[ None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=-2)
                /maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=-2))

    def delta_sigma_bary(self, rs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, r, params
        '''
        return np.moveaxis(
            projector.esd(rs, lambda r: self.rho_bary(r, mus, alphas, betas, gammas)) * (u.Msun/u.Mpc**2).to(self.units),
            0,
            1,
        )


@dataclass
class MatchingBaryonShearModel:
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    mass_definition: str = 'mean'
    delta: float = 200
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    shear_class: object = MatchingCmGnfwBaryonShear

    def __post_init__(self):
        self._shear = self.shear_class(
            zs=self.redshifts,
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
        )

    def normed_lensing_weights(self):
        return self.lensing_weights/self.lensing_weights.sum()

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]

    def delta_sigma_total(self, rs, alphas, betas, gammas, a_szs):
        mus = self.mu_from_sz_mu(np.log(self.sz_masses), a_szs).flatten()
        return self._shear.delta_sigma_total(rs, mus, alphas, betas, gammas)

    def delta_sigma(self, rs, alphas, betas, gammas, a_szs):
        'SHAPE cluster, a_sz, r, params'
        profiles = self.delta_sigma_total(rs, alphas, betas, gammas, a_szs)
        assert False, profiles.shape
        return (self.normed_lensing_weights()[None, :, None, None] * profiles)


@dataclass
class BaryonShearModel:
    mu_bins: np.ndarray
    redshift_bins: np.ndarray
    selection_func_file: object = maszcal.defaults.DefaultSelectionFunc()
    lensing_weights_file: object = maszcal.defaults.DefaultLensingWeights()
    cosmo_params: object = maszcal.defaults.DefaultCosmology()
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    delta: float = 200
    mass_definition: str = 'mean'
    sz_scatter: float = 0.2
    shear_class: object = GnfwBaryonShear

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self._shear = self.shear_class(
            zs=self.redshift_bins,
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
        )

    def _init_stacker(self):
        self.stacker = Stacker(
            mu_bins=self.mu_bins,
            redshift_bins=self.redshift_bins,
            cosmo_params=self.cosmo_params,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            comoving=self.comoving_radii,
            delta=self.delta,
            mass_definition=self.mass_definition,
            units=self.units,
            sz_scatter=self.sz_scatter,
        )

    def stacked_delta_sigma(self, rs, cons, alphas, betas, gammas, a_szs):
        delta_sigmas = self._shear.delta_sigma_total(rs, self.mu_bins, cons, alphas, betas, gammas)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)

    def weak_lensing_avg_mass(self, a_szs):
        try:
            return self.stacker.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.weak_lensing_avg_mass(a_szs)


@dataclass
class BaryonCmShearModel(BaryonShearModel):
    shear_class: object = CmGnfwBaryonShear

    def stacked_delta_sigma(self, rs, alphas, betas, gammas, a_szs):
        delta_sigmas = self._shear.delta_sigma_total(rs, self.mu_bins, alphas, betas, gammas)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)


@dataclass
class SingleMassBaryonShearModel:
    redshifts: np.ndarray
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    delta: float = 200
    mass_definition: str = 'mean'
    cosmo_params: object = maszcal.defaults.DefaultCosmology()
    shear_class: object = SingleMassGnfwBaryonShear

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self._shear = self.shear_class(
            zs=self.redshifts,
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
        )

    def delta_sigma(self, rs, mus, cons, alphas, betas, gammas):
        return self._shear.delta_sigma_total(rs, mus, cons, alphas, betas, gammas)


class Stacker:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            cosmo_params=maszcal.defaults.DefaultCosmology(),
            selection_func_file=maszcal.defaults.DefaultSelectionFunc(),
            lensing_weights_file=maszcal.defaults.DefaultLensingWeights(),
            comoving=None,
            delta=None,
            mass_definition=None,
            units=None,
            sz_scatter=None,
            matter_power_class=maszcal.matter.Power,
    ):
        self.mu_szs = mu_bins
        self.mus = mu_bins
        self.zs = redshift_bins
        self.matter_power_class = matter_power_class

        if isinstance(cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        if isinstance(selection_func_file, maszcal.defaults.DefaultSelectionFunc):
            self.selection_func = self._default_selection_func
        else:
            self.selection_func = maszcal.ioutils.get_selection_func_interpolator(selection_func_file)

        if isinstance(lensing_weights_file, maszcal.defaults.DefaultLensingWeights):
            self.lensing_weights = self._default_lensing_weights
        else:
            self.lensing_weights = maszcal.ioutils.get_lensing_weights_interpolator(lensing_weights_file)

        if delta is not None:
            self.delta = delta
        else:
            raise ValueError('delta must be specified')

        if mass_definition is not None:
            self.mass_definition = mass_definition
        else:
            raise ValueError('mass_definition must be specified')

        if units is not None:
            self.units = units
        else:
            raise ValueError('units must be specified')

        if sz_scatter is not None:
            self.sz_scatter = sz_scatter
        else:
            raise ValueError('sz_scatter must be specified')

        if comoving is not None:
            self.comoving = comoving
        else:
            raise ValueError('comoving must be specified')

        self.b_sz = 1

        self.ks = np.logspace(-4, 1, 400)

        self.constants = Constants()
        self._CUTOFF_MASS = 2e14

    def _default_selection_func(self, mu_szs, zs):
        '''
        SHAPE mu_sz, z
        '''
        sel_func = np.ones((mu_szs.size, zs.size))

        low_mass_indices = np.where(mu_szs < np.log(self._CUTOFF_MASS))
        sel_func[low_mass_indices, :] = 0

        return sel_func

    def _default_lensing_weights(self, zs):
        '''
        SHAPE mu, z
        '''
        return np.ones(zs.shape)

    def prob_musz_given_mu(self, mu_szs, mus, a_szs):
        '''
        SHAPE mu_sz, mu, params
        '''
        pref = 1/(np.sqrt(2*np.pi) * self.sz_scatter)

        diff = (mu_szs[:, None] - mus[None, :])[..., None] - a_szs[None, None, :]

        exps = np.exp(-diff**2 / (2*(self.sz_scatter)**2))

        return pref*exps

    def mass_sz(self, mu_szs):
        return np.exp(mu_szs)

    def mass(self, mus):
        return np.exp(mus)

    def calc_power_spect(self):
        power = self.matter_power_class(cosmo_params=self.cosmo_params)
        self.power_spect = power.spectrum(self.ks, self.zs, is_nonlinear=False)

        if np.isnan(self.power_spect).any():
            raise ValueError('Power spectrum contains NaN values.')

    def _init_tinker_hmf(self):
        self.mass_func = TinkerHmf(
            delta=self.delta,
            mass_definition=self.mass_definition,
            astropy_cosmology=self.astropy_cosmology,
            comoving=self.comoving,
        )

    def dnumber_dlogmass(self):
        '''
        SHAPE mu, z
        '''
        masses = self.mass(self.mus)

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        try:
            dn_dlogms = self.mass_func.dn_dlnm(masses, self.zs, self.ks, power_spect)
        except AttributeError:
            self._init_tinker_hmf()
            dn_dlogms = self.mass_func.dn_dlnm(masses, self.zs, self.ks, power_spect)

        if np.isnan(dn_dlogms).any():
            raise ValueError('Mass function has returned NaN values.')

        return dn_dlogms

    def comoving_vol(self):
        '''
        SHAPE z
        '''
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs).value
        hubble_z = self.astropy_cosmology.H(self.zs).value

        return c * comov_dist**2 / hubble_z

    def _sz_measure(self, a_szs):
        '''
        SHAPE mu_sz, mu, z, params
        '''
        return (self.mass_sz(self.mu_szs)[:, None, None, None]
                * self.selection_func(self.mu_szs, self.zs)[:, None, :, None]
                * self.prob_musz_given_mu(self.mu_szs, self.mus, a_szs)[:, :, None, :])

    def number_sz(self, a_szs):
        '''
        SHAPE params
        '''
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(self._sz_measure(a_szs), axis=0, dx=dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(self.dnumber_dlogmass()[..., None] * mu_sz_integral, axis=0, dx=dmus)

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral

    def stacked_delta_sigma(self, delta_sigmas, rs, a_szs):
        '''
        SHAPE r, params
        '''
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            (self._sz_measure(a_szs)[:, :, :, None, :]
             * delta_sigmas[None, ...]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def weak_lensing_avg_mass(self, a_szs):
        mass_wl = self.mass(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * mass_wl[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Weak lensing average masses contain NaN values.')

        return z_integral/self.number_sz(a_szs)


class CmStacker(Stacker):
    '''
    Changes a method to allow use of a con-mass relation following Miyatake et al 2019
    '''
    def stacked_delta_sigma(self, delta_sigmas, rs, a_szs):
        '''
        SHAPE r, params
        '''
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            (self._sz_measure(a_szs)[:, :, :, None, :]
             * delta_sigmas[None, ..., None]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def weak_lensing_avg_mass(self, a_szs):
        masses = self.mass(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * masses[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Weak lensing average masses contain NaN values.')

        return z_integral/self.number_sz(a_szs)


class MiyatakeStacker(Stacker):

    '''
    Changes a method to allow use of a con-mass relation following Miyatake et al 2019
    '''
    def stacked_delta_sigma(self, delta_sigmas, rs, a_szs):
        '''
        SHAPE r, params
        '''
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            (self._sz_measure(a_szs)[:, :, :, None, :]
             * delta_sigmas[None, ..., None]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = ConModel(mass_def, cosmology=self.cosmo_params)

    def _m500c(self, mus):
        masses = self.mass(mus)
        mass_def = str(self.delta) + self.mass_definition[0]

        try:
            masses_500c = self._con_model.convert_mass_def(masses, self.zs, mass_def, '500c')
        except AttributeError:
            self._init_con_model()
            masses_500c = self._con_model.convert_mass_def(masses, self.zs, mass_def, '500c')

        return masses_500c

    def weak_lensing_avg_mass(self, a_szs):
        masses_500c = self._m500c(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * masses_500c[None, :, :, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mus)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Weak lensing average masses contain NaN values.')

        return z_integral/self.number_sz(a_szs)


class NfwShearModel:
    '''
    Canonical variable order:
    mu_sz, mu, z, r, params
    '''
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            selection_func_file=maszcal.defaults.DefaultSelectionFunc(),
            lensing_weights_file=maszcal.defaults.DefaultLensingWeights(),
            cosmo_params=maszcal.defaults.DefaultCosmology(),
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            sz_scatter=0.2,
    ):

        if isinstance(cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.selection_func_file = selection_func_file
        self.lensing_weights_file = lensing_weights_file

        self.mu_szs = mu_bins
        self.mus = mu_bins
        self.zs = redshift_bins

        self.delta = delta
        self.mass_definition = mass_definition

        self.units = units
        self.comoving_radii = comoving_radii

        self.sz_scatter = sz_scatter

    def _init_nfw(self):
        self.nfw_model = maszcal.nfw.NfwModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def _init_stacker(self):
        self.stacker = Stacker(
            mu_bins=self.mus,
            redshift_bins=self.zs,
            cosmo_params=self.cosmo_params,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            comoving=self.comoving_radii,
            delta=self.delta,
            mass_definition=self.mass_definition,
            units=self.units,
            sz_scatter=self.sz_scatter,
        )

    def mass(self, mus):
        return np.exp(mus)

    def delta_sigma(self, rs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)

    def stacked_delta_sigma(self, rs, cons, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mus, cons)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)

    def weak_lensing_avg_mass(self, a_szs):
        try:
            return self.stacker.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.weak_lensing_avg_mass(a_szs)


class NfwCmShearModel(NfwShearModel):
    def _init_stacker(self):
        self.stacker = CmStacker(
            mu_bins=self.mus,
            redshift_bins=self.zs,
            cosmo_params=self.cosmo_params,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            comoving=self.comoving_radii,
            delta=self.delta,
            mass_definition=self.mass_definition,
            units=self.units,
            sz_scatter=self.sz_scatter,
        )

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = ConModel(mass_def, cosmology=self.cosmo_params)

    def _con(self, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, self.zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, self.zs, mass_def)

    def _init_nfw(self):
        self.nfw_model = maszcal.nfw.NfwCmModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def delta_sigma(self, rs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.zs, masses, self._con(masses))
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.zs, masses, self._con(masses))

    def stacked_delta_sigma(self, rs, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mus)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)


class MiyatakeShearModel(NfwShearModel):
    '''
    Changes some methods to enable use of a concentration-mass relation
    '''
    def _init_stacker(self):
        self.stacker = MiyatakeStacker(
            mu_bins=self.mus,
            redshift_bins=self.zs,
            cosmo_params=self.cosmo_params,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            comoving=self.comoving_radii,
            delta=self.delta,
            mass_definition=self.mass_definition,
            units=self.units,
            sz_scatter=self.sz_scatter,
        )

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = ConModel(mass_def, cosmology=self.cosmo_params)

    def _con(self, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, self.zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, self.zs, mass_def)

    def _init_nfw(self):
        self.nfw_model = maszcal.nfw.NfwCmModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def delta_sigma(self, rs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.zs, masses, self._con(masses))
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.zs, masses, self._con(masses))

    def stacked_delta_sigma(self, rs, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mus)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)


class SingleMassNfwShearModel:
    def __init__(
            self,
            redshifts,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=maszcal.defaults.DefaultCosmology(),
    ):

        self.redshifts = redshifts
        self.units = units
        self.comoving_radii = comoving_radii
        self.delta = delta
        self.mass_definition = mass_definition

        if isinstance(cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def _init_nfw(self):
        self.nfw_model = maszcal.nfw.NfwModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def mass(self, mu):
        return np.exp(mu)

    def delta_sigma(self, rs, mus, concentrations):
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.redshifts, masses, concentrations)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshifts, masses, concentrations)
