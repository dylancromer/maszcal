import camb

def get_camb_params(cosmology_params):
    params = camb.model.CAMBparams()

    params.set_cosmology(
        H0 = cosmology_params.hubble_constant,
        ombh2 = cosmology_params.omega_bary_hsqr,
        omch2 = cosmology_params.omega_cdm_hsqr,
        omk = 0,
        mnu = cosmology_params.neutrino_mass_sum,
        tau = cosmology_params.tau_reion,
    )

    return params
