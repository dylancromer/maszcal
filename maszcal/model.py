import numpy as np
import scipy.integrate as integrate
import camb
from offset_nfw.nfw import NFWModel
from astropy import units as u
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.nfw import SimpleDeltaSigma
from maszcal.likelihood import GaussianLikelihood




class DefaultCosmology():
    pass

class NoPowerSpectrum():
    pass

class StackedModel():
    def __init__(self,
                 cosmo_params=DefaultCosmology(),
                 power_spectrum=NoPowerSpectrum()):

        ### FITTING PARAMETERS AND LIKELIHOOD ###
        self.sigma_muszmu = 0.2
        self.a_sz = 0
        self.b_sz = 1
        self.a_wl = 0
        self.b_wl = 1
        self.concen_param = 2

        self.likelihood = GaussianLikelihood()

        ### SPATIAL QUANTITIES AND MATTER POWER ###
        self.zs =  np.linspace(0, 2, 20)
        self.max_k = 10
        self.min_k = 1e-4
        self.number_ks = 400

        if isinstance(power_spectrum, NoPowerSpectrum):
            pass
        else:
            self.ks, self.power_spect = power_spectrum

        ### COSMOLOGICAL PARAMETERS ###
        if isinstance(cosmo_params, DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)


        ### CLUSTER MASSES AND RELATED ###
        self.mu_szs = np.linspace(12, 16, 20)
        self.mus = np.linspace(12, 16, 20)
        self.concentrations = self.concen_param * np.ones(20)

        ### MISC CONSTANTS ###
        self.constants = Constants()


    def calc_power_spect(self):
        params = get_camb_params(self.cosmo_params, self.max_k, self.zs)

        results = camb.get_results(params)

        self.ks, _, self.power_spect = results.get_matter_power_spectrum(minkh = self.min_k,
                                                                         maxkh = self.max_k,
                                                                         npoints = self.number_ks)


    def init_onfw(self):
        self.onfw_model = NFWModel(self.astropy_cosmology)



    def mu_sz(self, mus):
        return self.b_sz*mus + self.a_sz


    def mu_wl(self, mus):
        return self.b_wl*mus + self.a_wl


    def prob_musz_given_mu(self, mu_szs, mus):
        mu_szs = mu_szs[np.newaxis, :]
        mus = mus[:, np.newaxis]

        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        exps = np.exp(-(mu_szs - mus - self.a_sz)**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps


    def mass_sz(self, mu_szs):
        return 10**mu_szs


    def mass(self, mus):
        return 10**mus


    def selection_func(self, mu_szs):
        sel_func = np.ones((self.zs.size, mu_szs.size))

        low_mass_indices = np.where(mu_szs < np.log10(3e14))
        sel_func[:, low_mass_indices] = 0

        return sel_func


    def delta_sigma_of_mass_alt(self, rs, mus):
        rhocrit_of_z_func = lambda z: self.cosmo_params.rho_crit * self.astropy_cosmology.efunc(z)**2
        simple_delta_sig = SimpleDeltaSigma(self.cosmo_params, self.zs, rhocrit_of_z_func)

        return simple_delta_sig.delta_sigma_of_mass(rs, mus, 200) #delta=200


    def delta_sigma_of_mass(self, rs, mus, concentrations=None, units=u.Msun/u.pc**2):
        masses = self.mass(mus)

        if concentrations is None:
            concentrations = self.concentrations

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs).to(units)
            return result.value.T
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs).to(units)
            return result.value.T


    def dnumber_dlogmass(self):
        masses = self.mass(self.mus)
        overdensity = 200
        rho_matter = self.cosmo_params.rho_crit * self.cosmo_params.omega_matter / self.cosmo_params.h**2

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        dn_dlogms = dn_dlogM(masses, self.zs, rho_matter, overdensity, self.ks, power_spect, 'comoving')
        return dn_dlogms.T


    def lensing_weights(self):
        return np.ones(self.zs.shape)


    def comoving_vol(self):
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs)
        hubble_z = self.astropy_cosmology.H(self.zs)

        return c * comov_dist**2 / hubble_z


    def _sz_measure(self):
        #TODO: maybe make this an @property and save the result???
        return (self.mass_sz(self.mu_szs)[np.newaxis, np.newaxis, np.newaxis, :]
                * self.selection_func(self.mu_szs)[np.newaxis, :, np.newaxis, :]
                * self.prob_musz_given_mu(self.mu_szs, self.mus)[np.newaxis, np.newaxis, :, :])


    def number_sz(self):
        #TODO: maybe make this an @property and save the result???
        mu_sz_integrand = self._sz_measure()
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dlogmass()[np.newaxis, :, :]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        return z_integral


    def delta_sigma(self, rs):
        normalization = 1/self.number_sz()

        mu_sz_integrand = (self._sz_measure()
                           * self.delta_sigma_of_mass(rs, self.mus, self.concentrations)[:, np.newaxis, :, np.newaxis])

        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dlogmass()[np.newaxis, :, :]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        delta_sigmas = normalization * z_integral
        return delta_sigmas


    def weak_lensing_avg_mass(self):
        normalization = 1/self.number_sz()

        mu_wl = self.mu_wl(self.mus)
        mass_wl = self.mass(mu_wl)[np.newaxis, np.newaxis, :, np.newaxis]

        mu_sz_integrand = self._sz_measure() * mass_wl
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dlogmass()[np.newaxis, :, :]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        avg_wl_mass = normalization * z_integral
        return avg_wl_mass
