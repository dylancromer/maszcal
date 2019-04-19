from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_camb_params


cosmology = CosmoParams()

def test_get_camb_params():
    params = get_camb_params(cosmology)

    assert params is not None
