from dataclasses import dataclass
import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as const
import maszcal.cosmo_utils


class NonFlatUniverseError(Exception):
    pass


class MatterInconsistencyError(Exception):
    pass


class HubbleConstantError(Exception):
    pass


@dataclass
class CosmoParams:
    hubble_constant: float = Planck15.H0.value
    omega_bary_hsqr: float = Planck15.Ob0*Planck15.h**2
    omega_cdm_hsqr: float = Planck15.Odm0*Planck15.h**2
    spectral_index: float = 0.9667
    scalar_amp: float = 2.2e-9
    sigma_8: float = 0.830
    tau_reion: float = 0.06
    omega_bary: float = Planck15.Ob0
    omega_cdm: float = Planck15.Odm0
    omega_matter: float = Planck15.Om0
    omega_lambda: float = Planck15.Ode0
    rho_crit: float = Planck15.critical_density(0).to(u.Msun/u.Mpc**3).value

    h: float = Planck15.h
    cmb_temp: float = Planck15.Tcmb0.value

    w0: float = -1.0
    wa: float = 0.0
    neutrino_mass_sum: float = 0.06

    use_ppf: bool = True
    flat: bool = True

    def _check_flatness(self):
        if not np.allclose(self.omega_matter + self.omega_lambda, 1, rtol=1e-2):
            raise NonFlatUniverseError('omega_matter and omega_lambda must sum to 1')

    def _check_matter_consistency(self):
        if not np.allclose(self.omega_matter, self.omega_cdm + self.omega_bary):
            raise MatterInconsistencyError('omega_cdm and omega_bary must sum to omega_matter')

    def _check_hsqr_params(self):
        if not np.allclose(self.omega_bary, self.omega_bary_hsqr/self.h**2):
            raise MatterInconsistencyError('omega_bary_hsqr must equal omega_bary * h**2')
        if not np.allclose(self.omega_cdm, self.omega_cdm_hsqr/self.h**2):
            raise MatterInconsistencyError('omega_cdm_hsqr must equal omega_cdm * h**2')

    def _check_hubble_consistency(self):
        if not np.allclose(self.hubble_constant/100, self.h):
            raise HubbleConstantError('hubble_constant must equal h*100')

    def __post_init__(self):
        if self.flat:
            self._check_flatness()
        self._check_matter_consistency()
        self._check_hsqr_params()
        self._check_hubble_consistency()


@dataclass
class Constants:
    speed_of_light: float = 2.99792e5  # km/s


@dataclass
class SigmaCrit:
    cosmo_params: object
    comoving: bool
    units: object = u.Msun/u.pc**2

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _check_zs(self, z_source, z_lens):
        if np.any(z_source < z_lens):
            raise ValueError('Source redshift must be greater or equal to lens redshift')

    def prefac(self):
        return const.c**2 / (4 * np.pi * const.G)

    def comoving_distance_z1z2(self, z1, z2):
        return self.astropy_cosmology.comoving_distance(z2) - self.astropy_cosmology.comoving_distance(z1)

    def distance_to(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z)

    def distance_between(self, z1, z2):
        return self.astropy_cosmology.angular_diameter_distance_z1z2(z1, z2)

    def sdc(self, z_source, z_lens):
        self._check_zs(z_source, z_lens)
        d_source = self.distance_to(z_source)
        d_lens = self.distance_to(z_lens)
        d_lens_source = self.distance_between(z_lens, z_source)
        surface_dens_crit = (self.prefac() * d_source / (d_lens * d_lens_source)).to(self.units).value
        if self.comoving:
            surface_dens_crit = surface_dens_crit / (1+z_lens)**2
        return surface_dens_crit
