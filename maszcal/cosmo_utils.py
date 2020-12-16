import numpy as np
from astropy.cosmology import Flatw0waCDM
import astropy.units as u
import camb


def get_camb_params(cosmology_params, max_k, zs, nonlinear_matter_power):
    params = camb.model.CAMBparams()

    params.set_cosmology(
        H0=cosmology_params.hubble_constant,
        ombh2=cosmology_params.omega_bary_hsqr,
        omch2=cosmology_params.omega_cdm_hsqr,
        omk=0,
        mnu=cosmology_params.neutrino_mass_sum,
        tau=cosmology_params.tau_reion,
    )

    if cosmology_params.use_ppf:
        de_model = 'ppf'
    else:
        de_model = 'fluid'

    params.set_dark_energy(
        w=cosmology_params.w0,
        cs2=1.0,
        wa=cosmology_params.wa,
        dark_energy_model=de_model
    )

    params.InitPower.set_params(
        As=cosmology_params.scalar_amp,
        ns=cosmology_params.spectral_index
    )

    params.set_matter_power(
        redshifts=zs,
        kmax=max_k,
        k_per_logint=None,
        nonlinear=nonlinear_matter_power,
        accurate_massive_neutrino_transfers=False,
        silent=True
    )

    return params


def get_astropy_cosmology(cosmology_params):
    omega_bary = cosmology_params.omega_bary_hsqr/cosmology_params.h**2

    astropy_cosmology = Flatw0waCDM(
        H0=cosmology_params.hubble_constant,
        Om0=cosmology_params.omega_matter,
        w0=cosmology_params.w0,
        wa=cosmology_params.wa,
        Tcmb0=cosmology_params.cmb_temp,
        m_nu=[0, 0, cosmology_params.neutrino_mass_sum] * u.eV,
        Ob0=omega_bary
    )

    return astropy_cosmology


def get_colossus_params(cosmology_params):
    omega_tot = cosmology_params.omega_matter + cosmology_params.omega_lambda
    is_flat = np.allclose(omega_tot, 1, rtol=3e-3)

    return {
        'flat': is_flat,
        'H0': cosmology_params.hubble_constant,
        'Om0': cosmology_params.omega_matter,
        'Ob0': cosmology_params.omega_bary,
        'sigma8': cosmology_params.sigma_8,
        'ns': cosmology_params.spectral_index,
    }
