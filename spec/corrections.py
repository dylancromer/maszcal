import numpy as np
import pytest
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology
import maszcal.corrections


def fake_1_halo_func(rs, zs, mus, *params):
    return np.ones(rs.shape + zs.shape + (params[0].size,))


def describe_Matching2HaloCorrection():

    def describe_stacked_excess_surface_density():

        @pytest.fixture
        def model():
            rs = np.logspace(-1, 1, 8)
            def fake_2_halo_func(zs, mus): return 1001*np.ones(mus.shape + rs.shape)

            return maszcal.corrections.Matching2HaloCorrection(
                radii=rs,
                one_halo_func=fake_1_halo_func,
                two_halo_func=fake_2_halo_func,
            )

        def it_combines_one_and_two_halo_profiles(model):
            mus = np.linspace(31, 33, 5)
            zs = np.linspace(0.1, 0.5, 5)
            one_halo_params = np.stack([np.arange(3), np.arange(3)])
            a_2hs = np.arange(2)

            esds = model.excess_surface_density(a_2hs, zs, mus, *one_halo_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 5, 3, 2)
            assert np.all(esds[..., 1] > 1000)
