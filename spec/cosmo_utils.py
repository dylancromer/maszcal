from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
import numpy as np


cosmology = CosmoParams()
max_k = 1
zs = np.linspace(0,1,2)


def test_get_camb_params():
    params = get_camb_params(cosmology, max_k, zs)

    assert params.H0 is not None

def test_get_astropy_cosmology():
    astropy_cosmo = get_astropy_cosmology(cosmology)

    assert astropy_cosmo.m_nu is not None
