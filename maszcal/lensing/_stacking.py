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
import maszcal.lensing._core as _core


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
    shear_class: object = _core.GnfwBaryonShear
    esd_func: object = projector.esd

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.NfwModel,
            esd_func=self.esd_func,
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
        delta_sigmas = self._shear.delta_sigma_total(rs, self.redshift_bins, self.mu_bins, cons, alphas, betas, gammas)

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
    shear_class: object = _core.CmGnfwBaryonShear
    con_class: object = ConModel

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.NfwCmModel,
            con_class=self.con_class,
            esd_func=self.esd_func,
        )

    def stacked_delta_sigma(self, rs, alphas, betas, gammas, a_szs):
        delta_sigmas = self._shear.delta_sigma_total(rs, self.redshift_bins, self.mu_bins, alphas, betas, gammas)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)


@dataclass
class Stacker:
    b_sz = 1
    ks = np.logspace(-4, 1, 400)
    constants = Constants()
    _CUTOFF_MASS = 2e14

    mu_bins: np.ndarray
    redshift_bins: np.ndarray
    cosmo_params: object = maszcal.defaults.DefaultCosmology()
    selection_func_file: object = maszcal.defaults.DefaultSelectionFunc()
    lensing_weights_file: object = maszcal.defaults.DefaultLensingWeights()
    comoving: bool = None
    delta: int = None
    mass_definition: str = None
    units: object = None
    sz_scatter: float = None
    matter_power_class: object = maszcal.matter.Power

    def __post_init__(
            self,
    ):
        self.mu_szs = self.mu_bins

        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        if isinstance(self.selection_func_file, maszcal.defaults.DefaultSelectionFunc):
            self.selection_func = self._default_selection_func
        else:
            self.selection_func = maszcal.ioutils.get_selection_func_interpolator(selection_func_file)

        if isinstance(self.lensing_weights_file, maszcal.defaults.DefaultLensingWeights):
            self.lensing_weights = self._default_lensing_weights
        else:
            self.lensing_weights = maszcal.ioutils.get_lensing_weights_interpolator(lensing_weights_file)

        if self.delta is None:
            raise ValueError('delta must be specified')

        if self.mass_definition is None:
            raise ValueError('mass_definition must be specified')

        if self.units is None:
            raise ValueError('units must be specified')

        if self.sz_scatter is None:
            raise ValueError('sz_scatter must be specified')

        if self.comoving is None:
            raise ValueError('comoving must be specified')


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
        self.power_spect = power.spectrum(self.ks, self.redshift_bins, is_nonlinear=False)

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
        masses = self.mass(self.mu_bins)

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        try:
            dn_dlogms = self.mass_func.dn_dlnm(masses, self.redshift_bins, self.ks, power_spect)
        except AttributeError:
            self._init_tinker_hmf()
            dn_dlogms = self.mass_func.dn_dlnm(masses, self.redshift_bins, self.ks, power_spect)

        if np.isnan(dn_dlogms).any():
            raise ValueError('Mass function has returned NaN values.')

        return dn_dlogms

    def comoving_vol(self):
        '''
        SHAPE z
        '''
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.redshift_bins).value
        hubble_z = self.astropy_cosmology.H(self.redshift_bins).value

        return c * comov_dist**2 / hubble_z

    def _sz_measure(self, a_szs):
        '''
        SHAPE mu_sz, mu, z, params
        '''
        return (self.mass_sz(self.mu_szs)[:, None, None, None]
                * self.selection_func(self.mu_szs, self.redshift_bins)[:, None, :, None]
                * self.prob_musz_given_mu(self.mu_szs, self.mu_bins, a_szs)[:, :, None, :])

    def number_sz(self, a_szs):
        '''
        SHAPE params
        '''
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(self._sz_measure(a_szs), axis=0, dx=dmu_szs)

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(self.dnumber_dlogmass()[..., None] * mu_sz_integral, axis=0, dx=dmus)

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
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

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def weak_lensing_avg_mass(self, a_szs):
        mass_wl = self.mass(self.mu_bins)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * mass_wl[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
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

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def weak_lensing_avg_mass(self, a_szs):
        masses = self.mass(self.mu_bins)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * masses[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Weak lensing average masses contain NaN values.')

        return z_integral/self.number_sz(a_szs)


class DefaultConClass:
    pass


@dataclass
class MiyatakeStacker(Stacker):
    con_class: object = DefaultConClass()
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

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Stacked delta sigmas contain NaN values.')

        return z_integral/self.number_sz(a_szs)[None, :]

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def _m500c(self, mus):
        masses = self.mass(mus)
        mass_def = str(self.delta) + self.mass_definition[0]

        try:
            masses_500c = self._con_model.convert_mass_def(masses, self.redshift_bins, mass_def, '500c')
        except AttributeError:
            self._init_con_model()
            masses_500c = self._con_model.convert_mass_def(masses, self.redshift_bins, mass_def, '500c')

        return masses_500c

    def weak_lensing_avg_mass(self, a_szs):
        masses_500c = self._m500c(self.mu_bins)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = maszcal.mathutils.trapz_(
            self._sz_measure(a_szs) * masses_500c[None, :, :, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mu_bins)
        mu_integral = maszcal.mathutils.trapz_(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.redshift_bins)
        z_integral = maszcal.mathutils.trapz_(
            ((
                self.lensing_weights(self.redshift_bins) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        if np.isnan(z_integral).any():
            raise ValueError('Weak lensing average masses contain NaN values.')

        return z_integral/self.number_sz(a_szs)


@dataclass
class NfwShearModel:
    '''
    Canonical variable order:
    mu_sz, mu, z, r, params
    '''
    mu_bins: np.ndarray
    redshift_bins: np.ndarray
    selection_func_file: str = maszcal.defaults.DefaultSelectionFunc()
    lensing_weights_file: str  = maszcal.defaults.DefaultLensingWeights()
    cosmo_params: object = maszcal.defaults.DefaultCosmology()
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    delta: int = 200
    mass_definition: str = 'mean'
    sz_scatter: float = 0.2

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

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

    def mass(self, mus):
        return np.exp(mus)

    def delta_sigma(self, rs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, cons)

    def stacked_delta_sigma(self, rs, cons, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mu_bins, cons)

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
class NfwCmShearModel(NfwShearModel):
    con_class: object = ConModel

    def _init_stacker(self):
        self.stacker = CmStacker(
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

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def _con(self, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, self.redshift_bins, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, self.redshift_bins, mass_def)

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
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, self._con(masses))
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, self._con(masses))

    def stacked_delta_sigma(self, rs, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mu_bins)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)


@dataclass
class MiyatakeShearModel(NfwShearModel):
    con_class: object = ConModel
    '''
    Changes some methods to enable use of a concentration-mass relation
    '''
    def _init_stacker(self):
        self.stacker = MiyatakeStacker(
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
            con_class=self.con_class,
        )

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def _con(self, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, self.redshift_bins, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, self.redshift_bins, mass_def)

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
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, self._con(masses))
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshift_bins, masses, self._con(masses))

    def stacked_delta_sigma(self, rs, a_szs):
        delta_sigmas = self.delta_sigma(rs, self.mu_bins)

        try:
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.stacked_delta_sigma(delta_sigmas, rs, a_szs)
