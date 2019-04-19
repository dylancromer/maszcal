from dataclasses import dataclass

@dataclass
class CosmoParams:
    hubble_constant: float = 67.74
    omega_bary_hsqr: float = 0.02230
    omega_cdm_hsqr: float = 0.1188
    spectral_index: float = 0.9667
    tau_reion: float = 0.066
    neutrino_mass_sum: float = 0.06
    omega_matter: float = 0.3089
    omega_lambda: float = 0.6911
    rho_crit: float = 1.274e11 #8.62e-27

    h: float = hubble_constant/100
