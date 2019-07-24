from dataclasses import dataclass


@dataclass
class CosmoParams:
    hubble_constant: float = 67
    omega_bary_hsqr: float = 0.022
    omega_cdm_hsqr: float = 0.1194
    spectral_index: float = 0.9667
    scalar_amp: float = 2.2e-9
    tau_reion: float = 0.06
    omega_matter: float = 0.3089
    omega_lambda: float = 0.6911
    rho_crit: float = 1.274e11 #8.62e-27 from wiki

    h: float = hubble_constant/100
    cmb_temp: float = 2.725

    w0: float = -1.0
    wa: float = 0.0
    neutrino_mass_sum: float = 0.06

    use_ppf: bool = True


@dataclass
class Constants:
    speed_of_light: float = 2.99792e5 # km/s
