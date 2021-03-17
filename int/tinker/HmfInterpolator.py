
from dataclasses import dataclass
import pytest
import numpy as np
import maszcal.cosmology
import maszcal.cosmo_utils
import maszcal.tinker
import maszcal.matter


def describe_HmfInterpolator():

    @pytest.fixture
    def hmf_interp():
        return maszcal.tinker.HmfInterpolator(
            mu_samples=np.log(np.geomspace(1e12, 1e16, 600)),
            redshift_samples=np.linspace(0.01, 4, 120),
            delta=200,
            mass_definition='mean',
            cosmo_params=maszcal.cosmology.CosmoParams(),
        )

    @pytest.fixture
    def mass_func(hmf_interp):
        delta = 200
        mass_definition = 'mean'
        apycosmo = maszcal.cosmo_utils.get_astropy_cosmology(hmf_interp.cosmo_params)
        return maszcal.tinker.TinkerHmf(delta, mass_definition, astropy_cosmology=apycosmo, comoving=True)

    def it_interpolates_the_cm_relation(hmf_interp, mass_func):
        mus = np.log(np.geomspace(1e13, 1e15, 400))
        zs = np.linspace(0.03, 3.5, 124)
        dn_dmus_interp = hmf_interp(zs, mus)
        assert dn_dmus_interp.shape == mus.shape + zs.shape

        power_spect = maszcal.matter.Power(cosmo_params=hmf_interp.cosmo_params).spectrum(
            hmf_interp.KS,
            zs,
            is_nonlinear=False,
        )

        dn_dmus_true = mass_func.dn_dlnm(np.exp(mus), zs, hmf_interp.KS, power_spect)

        assert np.allclose(dn_dmus_interp, dn_dmus_true, rtol=1e-2)
