import numpy as np
import scipy.integrate as integrate
from offset_nfw.nfw import NFWModel
from astropy.cosmology import FlatLambdaCDM

class StackedModel():
    def __init__(self):
        self.sigma_muszmu = 0.2
        self.a_param = 0
        self.b_param = 1

        self.mu_szs = np.linspace(1, 10, 10)
        self.mus = np.linspace(1, 10, 10)
        self.zs = 0.1 * np.linspace(0, 2, 10)


    def init_nfw(self):
        self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        self.nfw_model = NFWModel(self.cosmology)


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


    def concentration_from_mass(self, masses):
        return masses


    def delta_sigma_of_mass(self, rs, mus):
        masses = self.mass(mus)
        concentrations = self.concentration_from_mass(masses)

        try:
            return self.nfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)
        except AttributeError:
            self.init_nfw()
            return self.nfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)


    def dnumber_dmass(self):
        return np.ones(self.zs.shape)


    def lensing_weights(self):
        return np.ones(self.zs.shape)


    def comoving_vol(self):
        return np.ones(self.zs.shape)


    def number_sz(self):
        mu_sz_integrand = (self.mass_sz(self.mu_szs)[np.newaxis, np.newaxis, np.newaxis, :]
                           * self.selection_func(self.mu_szs)[np.newaxis, :, np.newaxis, :]
                           * self.prob_musz_given_mu(self.mu_szs, self.mus)[np.newaxis, np.newaxis, :, :])
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dmass()[np.newaxis, :, np.newaxis] * self.mass(self.mus)[np.newaxis, np.newaxis, :]
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
                           * self.delta_sigma_of_mass(rs, self.mus)[:, :, :, np.newaxis])
        mu_sz_integral = integrate.simps(mu_sz_integrand, x=self.mu_szs, axis=3)

        mu_integrand = self.dnumber_dmass()[np.newaxis, :, np.newaxis] * self.mass(self.mus)[np.newaxis, np.newaxis, :]
        mu_integrand = mu_integrand * mu_sz_integral
        mu_integral = integrate.simps(mu_integrand, x=self.mus, axis=2)

        z_integrand = (self.lensing_weights() * self.comoving_vol())[np.newaxis, :]
        z_integrand = z_integrand * mu_integral
        z_integral = integrate.simps(z_integrand, x=self.zs, axis=1)

        delta_sigmas = normalization * z_integral
        return delta_sigmas
