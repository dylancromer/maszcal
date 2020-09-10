import numpy as np
import pytest
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology
import maszcal.corrections


def fake_matching_1_halo_func(rs, zs, mus, *params):
    return np.ones(rs.shape + zs.shape + (params[0].size,))


def fake_1_halo_func(rs, zs, mus, *params):
    return np.ones(rs.shape + mus.shape + zs.shape + (params[0].size,))


def fake_2_halo_func(rs, zs, mus):
    return 1001*np.ones(mus.shape + zs.shape + rs.shape)


def fake_matching_2_halo_func(rs, zs, mus):
    return 1001*np.ones(mus.shape + rs.shape)


def describe_Matching2HaloCorrection():

    def describe_corrected_profile():

        @pytest.fixture
        def model():
            return maszcal.corrections.Matching2HaloCorrection(
                one_halo_func=fake_matching_1_halo_func,
                two_halo_func=fake_matching_2_halo_func,
            )

        def it_combines_one_and_two_halo_profiles(model):
            mus = np.linspace(31, 33, 5)
            zs = np.linspace(0.1, 0.5, 5)
            one_halo_params = np.stack([np.arange(3), np.arange(3)])
            a_2hs = np.arange(3)
            rs = np.geomspace(1e-1, 20, 8)

            esds = model.corrected_profile(rs, zs, mus, a_2hs, *one_halo_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 5, 3)
            assert np.all(esds[..., 1] > 1000)


def describe_TwoHaloCorrection():

    def describe_corrected_profile():

        @pytest.fixture
        def model():
            return maszcal.corrections.TwoHaloCorrection(
                one_halo_func=fake_1_halo_func,
                two_halo_func=fake_2_halo_func,
            )

        def it_combines_one_and_two_halo_profiles(model):
            mus = np.linspace(31, 33, 5)
            zs = np.linspace(0.1, 0.5, 4)
            one_halo_params = np.stack([np.arange(3), np.arange(3)])
            a_2hs = np.arange(3)
            rs = np.geomspace(1e-1, 20, 8)

            esds = model.corrected_profile(rs, zs, mus, a_2hs, *one_halo_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 5, 4, 3)
            assert np.all(esds[..., 1] > 1000)
