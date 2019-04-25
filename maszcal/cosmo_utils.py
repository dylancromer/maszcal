import camb
from astropy.cosmology import Flatw0waCDM
import astropy.units as u



def get_camb_params(cosmology_params, max_k, zs):
    params = camb.model.CAMBparams()

    params.set_cosmology(
        H0 = cosmology_params.hubble_constant,
        ombh2 = cosmology_params.omega_bary_hsqr,
        omch2 = cosmology_params.omega_cdm_hsqr,
        omk = 0,
        mnu = cosmology_params.neutrino_mass_sum,
        tau = cosmology_params.tau_reion,
    )

    if cosmology_params.use_ppf:
        de_model = 'ppf'
    else:
        de_model = 'fluid'

    params.set_dark_energy(
        w = cosmology_params.w0,
        cs2 = 1.0,
        wa = cosmology_params.wa,
        dark_energy_model = de_model
    )

    params.InitPower.set_params(
        As=cosmology_params.scalar_amp,
        ns=cosmology_params.spectral_index
    )

    params.set_matter_power(
        redshifts=[0.0],
        kmax=max_k,
        k_per_logint=None,
        nonlinear=False,
        accurate_massive_neutrino_transfers=False,
        silent=True
    )

    return params


def get_astropy_cosmology(cosmology_params):
    omega_bary = cosmology_params.omega_bary_hsqr/cosmology_params.h**2
    mass_per_neutrino = cosmology_params.neutrino_mass_sum/3 * u.eV

    astropy_cosmology = Flatw0waCDM(
        H0 = cosmology_params.hubble_constant,
        Om0 = cosmology_params.omega_matter,
        w0 = cosmology_params.w0,
        wa = cosmology_params.wa,
        Tcmb0 = cosmology_params.cmb_temp,
        m_nu = mass_per_neutrino,
        Ob0 = omega_bary
    )

    return astropy_cosmology
