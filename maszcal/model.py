### HIGH LEVEL DEPENDENCIES ###
import numpy as np
import xarray as xa
import pandas as pd
import scipy.integrate as integrate
### MID LEVEL DEPENDCIES ###
import camb
from astropy import units as u
### LOW LEVEL DEPENDENCIES ###
from maszcal.offset_nfw.nfw import NFWModel
### IN-MODULE DEPENDCIES ###
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.nfw import SimpleDeltaSigma
from maszcal.likelihood import GaussianLikelihood




def _trapz(xarr, dim, dx=None):
    other_dims = np.array(xarr.dims)
    assert dim in other_dims
    other_dims = tuple(other_dims[other_dims != dim])
    new_dims = (dim,) + other_dims
    xarr = xarr.transpose(*new_dims)

    if dx is None:
        dx = xa.DataArray(np.ones(xarr.shape[0]), dims=(dim))
    else:
        dx = xa.DataArray(dx, dims=(dim))

    xarr = xarr * dx

    return xa.DataArray(
        0.5*(xarr.values[0, ...] + 2*xarr.values[1:-1,...].sum(axis=0) + xarr.values[-1,...]),
        dims=other_dims,
    )


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
        self.a_sz = 2
        self.b_sz = 1
        self.a_wl = 0
        self.b_wl = 1
        self.concentrations = xa.DataArray(np.array([2]), dims=('concentration'))

        self.likelihood = GaussianLikelihood()

        ### SPATIAL QUANTITIES AND MATTER POWER ###
        self.zs =  xa.DataArray(np.linspace(0, 2, 20), dims=('redshift'))
        self.max_k = 10
        self.min_k = 1e-4
        self.number_ks = 400

        if isinstance(power_spectrum, NoPowerSpectrum):
            pass
        else:
            ks, power_spect = power_spectrum

        ### COSMOLOGICAL PARAMETERS ###
        if isinstance(cosmo_params, DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)


        ### CLUSTER MASSES AND RELATED ###
        self.mu_szs = xa.DataArray(np.linspace(12, 16, 20), dims=('mu_sz'))
        self.mus = xa.DataArray(np.linspace(12, 16, 20), dims=('mu'))

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
        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        diff = (mu_szs - mus) - self.a_sz

        exps = np.exp(-diff**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps

    def mass_sz(self, mu_szs):
        return 10**mu_szs

    def mass(self, mus):
        return 10**mus

    def selection_func(self, mu_szs):
        sel_func = np.ones((self.zs.size, mu_szs.size))

        low_mass_indices = np.where(mu_szs < np.log10(3e14))
        sel_func[:, low_mass_indices] = 0

        sel_func = xa.DataArray(sel_func, dims=('redshift', 'mu_sz'))

        return sel_func

    def delta_sigma_of_mass_alt(self, rs, mus):
        rhocrit_of_z_func = lambda z: self.cosmo_params.rho_crit * self.astropy_cosmology.efunc(z)**2
        simple_delta_sig = SimpleDeltaSigma(self.cosmo_params, self.zs, rhocrit_of_z_func)

        return simple_delta_sig.delta_sigma_of_mass(rs, mus, 200) #delta=200

    def delta_sigma_of_mass(self, rs, mus, concentrations=None, units=u.Msun/u.pc**2):
        masses = self.mass(mus)

        if concentrations is None:
            concentrations = self.concentrations

        rs = xa.DataArray(rs, dims=('radius'))
        masses = xa.DataArray(masses, dims=('mu'))
        concentrations = xa.DataArray(concentrations, dims=('concentration'))
        zs = xa.DataArray(self.zs, dims=('redshift'))

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

    def dnumber_dlogmass(self):
        masses = self.mass(self.mus).values
        overdensity = 200
        rho_matter = self.cosmo_params.rho_crit * self.cosmo_params.omega_matter / self.cosmo_params.h**2

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        dn_dlogms = dn_dlogM(
            masses,
            self.zs.values,
            rho_matter,
            overdensity,
            self.ks,
            power_spect,
            'comoving'
        )

        return xa.DataArray(dn_dlogms.T, dims=('redshift', 'mu'))

    def lensing_weights(self):
        return xa.DataArray(np.ones(self.zs.shape), dims=('redshift'))

    def comoving_vol(self):
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs)
        hubble_z = self.astropy_cosmology.H(self.zs)

        return xa.DataArray(c * comov_dist**2 / hubble_z, dims=('redshift'))

    def _sz_measure(self):
        #TODO: maybe make this an @property and save the result???
        return (self.mass_sz(self.mu_szs)
                * self.selection_func(self.mu_szs)
                * self.prob_musz_given_mu(self.mu_szs, self.mus))

    def number_sz(self):
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(self._sz_measure(), 'mu_sz', dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass() * mu_sz_integral, 'mu', dmus)

        dzs = np.gradient(self.zs.values)
        z_integral = _trapz(self.lensing_weights() * self.comoving_vol() * mu_integral, 'redshift', dzs)

        return z_integral

    def delta_sigma(self, rs, units=u.Msun/u.Mpc**2):
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure() * self.delta_sigma_of_mass(rs,
                                                           self.mus,
                                                           self.concentrations,
                                                           units=units)
             ),
            'mu_sz',
            dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass() * mu_sz_integral, 'mu', dmus)

        dzs = np.gradient(self.zs.values)
        z_integral = _trapz(self.lensing_weights() * self.comoving_vol() * mu_integral, 'redshift', dzs)

        return z_integral/self.number_sz()

    def weak_lensing_avg_mass(self):
        mu_wl = self.mu_wl(self.mus)
        mass_wl = self.mass(mu_wl)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(self._sz_measure() * mass_wl, 'mu_sz', dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass() * mu_sz_integral, 'mu', dmus)

        dzs = np.gradient(self.zs.values)
        z_integral = _trapz(self.lensing_weights() * self.comoving_vol() * mu_integral, 'redshift', dzs)

        return z_integral/self.number_sz()
