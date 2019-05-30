### HIGH LEVEL DEPENDENCIES ###
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from numba import jit
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




def atleast_kd(array, k):
    array = np.asarray(array)
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)


def _trapz(arr, axis, dx=None):
    arr = np.moveaxis(arr, axis, 0)

    if dx is None:
        dx = np.ones(arr.shape[0])
    dx = atleast_kd(dx, arr.ndim)

    arr = dx*arr

    return 0.5*(arr[0, ...] + 2*arr[1:-1,...].sum(axis=0) + arr[-1,...])


class DefaultCosmology():
    pass


class NoPowerSpectrum():
    pass


class StackedModel():
    """
    Canonical variable order:
    mu_sz, mu, z, r, c, a_sz
    """
    def __init__(self,
                 cosmo_params=DefaultCosmology(),
                 power_spectrum=NoPowerSpectrum()):

        ### FITTING PARAMETERS AND LIKELIHOOD ###
        self.sigma_muszmu = 0.2
        self.a_sz = np.array([2])
        self.b_sz = 1
        self.concentrations = np.array([2])

        ### SPATIAL QUANTITIES AND MATTER POWER ###
        self.zs =  np.linspace(0, 2, 20)
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
        self.mu_szs = np.linspace(12, 16, 20)
        self.mus = np.linspace(12, 16, 20)

        ### MISC ###
        self.constants = Constants()
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

        self.ks, _, self.power_spect = results.get_matter_power_spectrum(minkh = self.min_k,
                                                                         maxkh = self.max_k,
                                                                         npoints = self.number_ks)

    def init_onfw(self):
        self.onfw_model = NFWModel(self.astropy_cosmology, comoving=self.comoving_radii)

    def mu_sz(self, mus):
        return self.b_sz*mus + self.a_sz

    def prob_musz_given_mu(self, mu_szs, mus):
        """
        SHAPE mu_sz, mu, a_sz
        """
        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        diff = (mu_szs[:, None] - mus[None, :])[..., None] - self.a_sz[None, None, :]

        exps = np.exp(-diff**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps

    def mass_sz(self, mu_szs):
        return 10**mu_szs

    def mass(self, mus):
        return 10**mus

    def selection_func(self, mu_szs):
        """
        SHAPE mu_sz, z
        """
        sel_func = np.ones((mu_szs.size, self.zs.size))

        low_mass_indices = np.where(mu_szs < np.log10(3e14))
        sel_func[:, low_mass_indices] = 0

        return sel_func

    def delta_sigma_of_mass_alt(self, rs, mus):
        #TODO: Update shape to conform to canonical order
        rhocrit_of_z_func = lambda z: self.cosmo_params.rho_crit * self.astropy_cosmology.efunc(z)**2
        simple_delta_sig = SimpleDeltaSigma(self.cosmo_params, self.zs, rhocrit_of_z_func)

        return simple_delta_sig.delta_sigma_of_mass(rs, mus, 200) #delta=200

    def delta_sigma_of_mass(self, rs, mus, concentrations=None, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, c
        """
        masses = self.mass(mus)

        if concentrations is None:
            concentrations = self.concentrations

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

    def dnumber_dlogmass(self):
        """
        SHAPE mu, z
        """
        masses = self.mass(self.mus)
        overdensity = 200
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

        return dn_dlogms.T

    def lensing_weights(self):
        """
        SHAPE z
        """
        return np.ones(self.zs.shape)

    def comoving_vol(self):
        """
        SHAPE z
        """
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs)
        hubble_z = self.astropy_cosmology.H(self.zs)

        return c * comov_dist**2 / hubble_z

    def _sz_measure(self):
        """
        SHAPE mu_sz, mu, z, a_sz
        """
        return (self.mass_sz(self.mu_szs)[:, None, None, None]
                * self.selection_func(self.mu_szs)[:, None, :, None]
                * self.prob_musz_given_mu(self.mu_szs, self.mus)[:, :, None, :])

    def number_sz(self):
        """
        SHAPE a_sz
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(self._sz_measure(), axis=0, dx=dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass()[..., None] * mu_sz_integral, axis=0, dx=dmus)

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((self.lensing_weights() * self.comoving_vol())[:, None]
             * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral

    def delta_sigma(self, rs, units=u.Msun/u.Mpc**2):
        """
        SHAPE r, c, a_sz
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure()[:, :, :, None, None, :]
             * self.delta_sigma_of_mass(
                 rs,
                 self.mus,
                 self.concentrations,
                 units=units
             )[None, ..., None]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None, None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((self.lensing_weights() * self.comoving_vol())[:, None, None, None]
             * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz()[None, None, :]

    def weak_lensing_avg_mass(self):
        mass_wl = self.mass(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            self._sz_measure() * mass_wl[None, :, None, None],
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
            ((self.lensing_weights() * self.comoving_vol())[:, None]
             * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz()[None, :]
