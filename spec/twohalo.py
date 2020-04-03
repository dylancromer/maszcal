import pytest
import numpy as np
import maszcal.twohalo
import maszcal.cosmology


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model():
        cosmo = maszcal.cosmology.CosmoParams()
        return maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 3)
        rs = np.logspace(-1, 1, 2)

        esds = two_halo_model.esd(rs, mus, zs)

        assert np.all(esds > 0)
        assert not np.any(np.isnan(esds))
        assert esds.shape == mus.shape + zs.shape + rs.shape
