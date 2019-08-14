from dataclasses import dataclass
from astropy.cosmology import Planck15
import astropy.units as u


@dataclass
class CosmoParams:
    hubble_constant: float = Planck15.H0.value
    omega_bary_hsqr: float = Planck15.Ob0*Planck15.h**2
    omega_cdm_hsqr: float = Planck15.Odm0*Planck15.h**2
    spectral_index: float = 0.9667
    scalar_amp: float = 2.2e-9
    tau_reion: float = 0.06
    omega_matter: float = Planck15.Odm0
    omega_lambda: float = Planck15.Ode0
    rho_crit: float = Planck15.critical_density(0).to(u.Msun/u.Mpc**3).value

    h: float = Planck15.h
    cmb_temp: float = Planck15.Tcmb0.value

    w0: float = -1.0
    wa: float = 0.0
    neutrino_mass_sum: float = 0.06

    use_ppf: bool = True


@dataclass
class Constants:
    speed_of_light: float = 2.99792e5  # km/s
