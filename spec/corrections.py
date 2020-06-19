import numpy as np
import pytest
import astropy.units as u
import maszcal.cosmology
import maszcal.corrections


def fake_projector_esd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


def describe_Matching2HaloBaryonShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model(mocker):
            mocker.patch('maszcal.corrections.projector.esd', new=fake_projector_esd)
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)

            rs = np.logspace(-1, 1, 8)
            def fake_2_halo_func(zs, mus): return 1000*np.ones(mus.shape + rs.shape)

            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.corrections.Matching2HaloBaryonShearModel(
                radii=rs,
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                two_halo_term_function=fake_2_halo_func,
                cosmo_params=cosmo_params,
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
            )

        def it_calculates_stacked_delta_sigma_profiles(model):
            cons = 2*np.ones(2)
            alphas = np.ones(2)
            betas = np.ones(2)
            gammas = np.ones(2)
            a_2hs = np.arange(2)
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_delta_sigma(cons, alphas, betas, gammas, a_2hs, a_szs)

            assert np.all(esds >= 0)
            assert esds.shape == (3, 8, 2)
            assert np.all(esds[:, :, 1] >= 1000)
