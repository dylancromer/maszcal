import numpy as np
import scipy.integrate as integrate
import camb
from astropy import units as u
from maszcal.nfw import NfwModel
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.mathutils import _trapz
from maszcal.nothing import NoParams
import maszcal.ioutils as ioutils
import maszcal.defaults as defaults


class Stacker:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            cosmo_params=defaults.DefaultCosmology(),
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            delta=None,
            units=None,
    ):
        self.mu_szs = mu_bins
        self.mus = mu_bins
        self.zs = redshift_bins

        if isinstance(cosmo_params, defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        if isinstance(selection_func_file, defaults.DefaultSelectionFunc):
            self.selection_func = self._default_selection_func
        else:
            self.selection_func = ioutils.get_selection_func_interpolator(selection_func_file)

        if isinstance(lensing_weights_file, defaults.DefaultLensingWeights):
            self.lensing_weights = self._default_lensing_weights
        else:
            self.lensing_weights = ioutils.get_lensing_weights_interpolator(lensing_weights_file)

        if delta is not None:
            self.delta = delta
        else:
            raise ValueError('delta must be specified')

        if units is not None:
            self.units = units
        else:
            raise ValueError('units must be specified')

        self.sigma_muszmu = 0.2
        self.b_sz = 1

        self.max_k = 10
        self.min_k = 1e-4
        self.number_ks = 400

        self.constants = Constants()
        self._CUTOFF_MASS = 2e14

    def _default_selection_func(self, mu_szs, zs):
        """
        SHAPE mu_sz, z
        """
        sel_func = np.ones((mu_szs.size, zs.size))

        low_mass_indices = np.where(mu_szs < np.log(self._CUTOFF_MASS))
        sel_func[low_mass_indices, :] = 0

        return sel_func

    def _default_lensing_weights(self, zs):
        """
        SHAPE mu, z
        """
        return np.ones(zs.shape)

    def prob_musz_given_mu(self, mu_szs, mus, a_szs):
        """
        SHAPE mu_sz, mu, params
        """
        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        diff = (mu_szs[:, None] - mus[None, :])[..., None] - a_szs[None, None, :]

        exps = np.exp(-diff**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps

    def mass_sz(self, mu_szs):
        return np.exp(mu_szs)

    def mass(self, mus):
        return np.exp(mus)

    def calc_power_spect(self):
        params = get_camb_params(self.cosmo_params, self.max_k, self.zs)

        results = camb.get_results(params)

        self.ks, _, self.power_spect = results.get_matter_power_spectrum(minkh=self.min_k,
                                                                         maxkh=self.max_k,
                                                                         npoints=self.number_ks)

    def dnumber_dlogmass(self):
        """
        SHAPE mu, z

        UNITS h/Mpc
        """
        masses = self.mass(self.mus)
        overdensity = self.delta
        rho_matter = self.cosmo_params.rho_crit * self.cosmo_params.omega_matter / self.cosmo_params.h**2

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        dn_dlogms = dn_dlogM(
            masses,
            self.zs,
            rho_matter,
            overdensity,
            self.ks,
            power_spect,
            comoving=True
        )

        return dn_dlogms

    def comoving_vol(self):
        """
        SHAPE z
        """
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs).value
        hubble_z = self.astropy_cosmology.H(self.zs).value

        return c * comov_dist**2 / hubble_z

    def _sz_measure(self, a_szs):
        """
        SHAPE mu_sz, mu, z, params
        """
        return (self.mass_sz(self.mu_szs)[:, None, None, None]
                * self.selection_func(self.mu_szs, self.zs)[:, None, :, None]
                * self.prob_musz_given_mu(self.mu_szs, self.mus, a_szs)[:, :, None, :])

    def number_sz(self, a_szs):
        """
        SHAPE params
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(self._sz_measure(a_szs), axis=0, dx=dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass()[..., None] * mu_sz_integral, axis=0, dx=dmus)

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral

    def delta_sigma(self, delta_sigmas_of_mass, rs, a_szs):
        """
        SHAPE r, params
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure(a_szs)[:, :, :, None, :]
             * delta_sigmas_of_mass[None, ...]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz(a_szs)[None, :]

    def weak_lensing_avg_mass(self, a_szs):
        mass_wl = self.mass(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            self._sz_measure(a_szs) * mass_wl[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz(a_szs)


class StackedModel:
    """
    Canonical variable order:
    mu_sz, mu, z, r, params
    """
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            cosmo_params=defaults.DefaultCosmology(),
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
    ):

        if isinstance(cosmo_params, defaults.DefaultCosmology):
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
        self._comoving_radii = comoving_radii

    @property
    def comoving_radii(self):
        return self._comoving_radii

    @comoving_radii.setter
    def comoving_radii(self, rs_are_comoving):
        self._comoving_radii = rs_are_comoving
        self._init_nfw()

    def _init_nfw(self):
        self.nfw_model = NfwModel(
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
            delta=self.delta,
            units=self.units,
        )

    def mass(self, mus):
        return np.exp(mus)

    def delta_sigma_of_mass(self, rs, mus, cons):
        """
        SHAPE mu, z, r, params
        """
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)

    def delta_sigma(self, rs, cons, a_szs):
        delta_sigmas_of_mass = self.delta_sigma_of_mass(rs, self.mus, cons)

        try:
            return self.stacker.delta_sigma(delta_sigmas_of_mass, rs, a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.delta_sigma(delta_sigmas_of_mass, rs, a_szs)

    def weak_lensing_avg_mass(self, a_szs):
        try:
            return self.stacker.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacker()
            return self.stacker.weak_lensing_avg_mass(a_szs)


class SingleMassModel:
    def __init__(
            self,
            redshift,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
    ):

        self.redshift = np.array([redshift])
        self.units = units
        self.comoving_radii = comoving_radii
        self.delta = delta
        self.mass_definition = mass_definition

        if isinstance(cosmo_params, defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def _init_nfw(self):
        self.nfw_model = NfwModel(
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
            return self.nfw_model.delta_sigma(rs, self.redshift, masses, concentrations)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshift, masses, concentrations)


class GaussianBaryonicModel:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            cosmo_params=defaults.DefaultCosmology(),
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
    ):
        if isinstance(cosmo_params, defaults.DefaultCosmology):
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
        self._comoving_radii = comoving_radii

    @property
    def comoving_radii(self):
        return self._comoving_radii

    @comoving_radii.setter
    def comoving_radii(self, rs_are_comoving):
        self._comoving_radii = rs_are_comoving
        self._init_nfw()

    def _init_nfw(self):
        self.nfw_model = NfwModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def mass(self, mu):
        return np.exp(mu)

    def delta_sigma_baryon(self, rs, mus, baryon_vars):
        """
        SHAPE mu, z, r, params
        """
        masses = self.mass(mus)

        prefac = masses/(1e12*np.pi)
        exponen = np.exp(-rs[:, None]**2/(2*baryon_vars[None, :]))
        postfac = (1 - exponen)/rs**2 - exponen/(2*baryon_vars[None, :])
        return prefac[:, None, None]*postfac[None, :, :]

    def delta_sigma_nfw(self, rs, mus, cons):
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.zs, masses, cons)
