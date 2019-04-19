import numpy as np
import scipy.integrate as integrate
import camb
from offset_nfw.nfw import NFWModel
from astropy.cosmology import FlatLambdaCDM
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params
from maszcal.cosmology import CosmoParams




class StackedModel():
    def __init__(self):
        self.sigma_muszmu = 0.2
        self.a_param = 0
        self.b_param = 1

        self.mu_szs = np.log(np.logspace(10, 16, 10))
        self.mus = np.log(np.logspace(10, 16, 10))
        self.concentrations = np.logspace(0, 1, 10)
        self.zs =  np.linspace(0, 2, 10)
        self.ks = np.logspace(-4, -1, 200)

        self.cosmo_params = CosmoParams()


    def calc_power_spect(self):
        params = get_camb_params(self.cosmo_params)
        self.power_spectrum_interp = camb.get_matter_power_interpolator(
            params,
            zs=self.zs,
            kmax=0.14,
            nonlinear=False,
        )


    def init_nfw(self):
        #TODO: Astropy cosmology is broken. Need get rid of this dep
        astropy_cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        self.nfw_model = NFWModel(astropy_cosmology)


    def mu_sz(self, mus):
        return self.b_param*mus + self.a_param


    def prob_musz_given_mu(self, mu_szs, mus):
        mu_szs = mu_szs[np.newaxis, :]
        mus = mus[:, np.newaxis]

        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        exps = np.exp(-(mu_szs - mus - self.a_param)**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps


    def mass_sz(self, mu_szs):
        return np.exp(mu_szs)


    def mass(self, mus):
        return np.exp(mus)


    def selection_func(self, mu_szs):
        return np.ones((self.zs.size, mu_szs.size))


    def delta_sigma_of_mass(self, rs, mus, cons):
        masses = self.mass(mus)
        concentrations = cons

        try:
            return self.nfw_model.deltasigma_theory(rs, masses, concentrations, self.zs).value.transpose()
        except AttributeError:
            self.init_nfw()
            return self.nfw_model.deltasigma_theory(rs, masses, concentrations, self.zs).value.transpose()


    def dnumber_dlogmass(self):
        masses = self.mass(self.mus)
        overdensity = 200
        rho_matter = self.cosmo_params.rho_crit * self.cosmo_params.omega_matter

        try:
            power_spect = self.power_spectrum_interp(self.zs, self.ks)
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spectrum_interp(self.zs, self.ks)

        dn_dlogms = dn_dlogM(masses, self.zs, rho_matter, overdensity, self.ks, power_spect, 'comoving')
        return dn_dlogms


    def lensing_weights(self):
        return np.ones(self.zs.shape)


    def comoving_vol(self):
        return np.ones(self.zs.shape)


    def number_sz(self):
        mu_sz_integrand = (self.mass_sz(self.mu_szs)[np.newaxis, np.newaxis, np.newaxis, :]
                           * self.selection_func(self.mu_szs)[np.newaxis, :, np.newaxis, :]
                           * self.prob_musz_given_mu(self.mu_szs, self.mus)[np.newaxis, np.newaxis, :, :])
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dlogmass()[np.newaxis, :, np.newaxis]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        return z_integral


    def delta_sigma(self, rs):
        normalization = 1/self.number_sz()

        mu_sz_integrand = (self.mass_sz(self.mu_szs)[np.newaxis, np.newaxis, np.newaxis, :]
                           * self.selection_func(self.mu_szs)[np.newaxis, :, np.newaxis, :]
                           * self.prob_musz_given_mu(self.mu_szs, self.mus)[np.newaxis, np.newaxis, :, :]
                           * self.delta_sigma_of_mass(rs, self.mus, self.concentrations)[:, np.newaxis, :, np.newaxis])
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dlogmass()[np.newaxis, :, np.newaxis]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        delta_sigmas = normalization * z_integral
        return delta_sigmas
