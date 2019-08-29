import json
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d, interp2d
import camb
from astropy import units as u
from maszcal.offset_nfw.nfw import NFWModel
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.mathutils import _trapz
from maszcal.nothing import NoParams
import maszcal.defaults as defaults


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
            delta=200,
            mass_definition='mean',
    ):

        ### FITTING PARAMETERS AND LIKELIHOOD ###
        self.sigma_muszmu = 0.2
        self.b_sz = 1

        ### SPATIAL QUANTITIES AND MATTER POWER ###
        self.max_k = 10
        self.min_k = 1e-4
        self.number_ks = 400

        ### COSMOLOGICAL PARAMETERS ###
        if isinstance(cosmo_params, defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        ### CLUSTER MASSES AND REDSHIFTS###
        self.mu_szs = mu_bins
        self.mus = mu_bins
        self.zs = redshift_bins

        self.delta = delta
        self.mass_definition = mass_definition

        ### SELECTION FUNCTION ###
        if isinstance(selection_func_file, defaults.DefaultSelectionFunc):
            self.selection_func = self._default_selection_func
        else:
            self.selection_func = self._get_selection_func_interpolator(selection_func_file)

        ### LENSING WEIGHTS ###
        if isinstance(lensing_weights_file, defaults.DefaultLensingWeights):
            self.lensing_weights = self._default_lensing_weights
        else:
            self.lensing_weights = self._get_lensing_weights_interpolator(lensing_weights_file)

        ### MISC ###
        self.constants = Constants()
        self.NUM_OFFSET_THETAS = 10
        self.NUM_OFFSET_RADII = 30
        self._comoving_radii = True

    @property
    def comoving_radii(self):
        return self._comoving_radii

    @comoving_radii.setter
    def comoving_radii(self, rs_are_comoving):
        self._comoving_radii = rs_are_comoving
        self.init_onfw()

    def calc_power_spect(self):
        params = get_camb_params(self.cosmo_params, self.max_k, self.zs)

        results = camb.get_results(params)

        self.ks, _, self.power_spect = results.get_matter_power_spectrum(minkh=self.min_k,
                                                                         maxkh=self.max_k,
                                                                         npoints=self.number_ks)

    def init_onfw(self):
        rho_dict = {'mean':'rho_m', 'crit':'rho_c'}

        self.onfw_model = NFWModel(
            self.astropy_cosmology,
            comoving=self.comoving_radii,
            delta=self.delta,
            rho=rho_dict[self.mass_definition],
        )

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

    def _get_selection_func_interpolator(self, selection_func_file):
        with open(selection_func_file, 'r') as json_file:
            selec_func_dict = json.load(json_file)

        mus = np.asarray(selec_func_dict['mus'])
        zs = np.asarray(selec_func_dict['zs'])
        selection_fs = np.asarray(selec_func_dict['selection_fs'])
        interpolator = interp2d(zs, mus, selection_fs, kind='linear')

        return lambda mu, z: interpolator(z, mu)

    def _default_selection_func(self, mu_szs, zs):
        """
        SHAPE mu_sz, z
        """
        sel_func = np.ones((mu_szs.size, zs.size))

        low_mass_indices = np.where(mu_szs < np.log(3e14))
        sel_func[low_mass_indices, :] = 0

        return sel_func

    def delta_sigma_of_mass(self, rs, mus, cons, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, params
        """
        masses = self.mass(mus)

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, cons, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, cons, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

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

    def _get_lensing_weights_interpolator(self, lensing_weights_file):
        with open(lensing_weights_file, 'r') as json_file:
            weights_dict = json.load(json_file)

        zs = np.asarray(weights_dict['zs'])
        weights = np.asarray(weights_dict['weights'])

        return interp1d(zs, weights, kind='cubic')

    def _default_lensing_weights(self, zs):
        """
        SHAPE mu, z
        """
        return np.ones(zs.shape)

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

    def delta_sigma(self, rs, cons, a_szs, units=u.Msun/u.pc**2):
        """
        SHAPE r, params
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure(a_szs)[:, :, :, None, :]
             * self.delta_sigma_of_mass(
                 rs,
                 self.mus,
                 cons,
                 units=units,
                 miscentered=False,
             )[None, ...]),
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


class SingleMassModel:
    def __init__(
            self,
            redshift,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
    ):

        self.redshift = np.array([redshift])
        self.comoving_radii = comoving_radii
        self.delta = delta
        self.mass_definition = mass_definition

        if isinstance(cosmo_params, defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def init_onfw(self):
        rho_dict = {'mean':'rho_m', 'crit':'rho_c'}

        self.onfw_model = NFWModel(
            self.astropy_cosmology,
            comoving=self.comoving_radii,
            delta=self.delta,
            rho=rho_dict[self.mass_definition],
        )

    def mass(self, mu):
        return np.exp(mu)

    def delta_sigma(self, rs, mus, concentrations, units=u.Msun/u.pc**2):

        masses = self.mass(mus)

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.redshift)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.redshift)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

