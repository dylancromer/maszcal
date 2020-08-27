from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology


def fake_projector_esd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


def fake_projector_sd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


def fake_rho_total(rs, zs, mus, *params):
    return np.ones(rs.shape + mus.shape + (params[0].size,))


def fake_scattered_rho_total(rs, zs, mus, *params):
    return np.ones(rs.shape + mus.shape + zs.shape + (params[0].size,))


def fake_logmass_prob(z, mu):
    return np.ones(mu.shape + z.shape)


def describe_ScatteredMatchingConvergenceModel():

    def describe_angle_scale_distance():

        @pytest.fixture
        def model_comoving():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_scattered_rho_total,
                comoving=True,
                logmass_prob_dist_func=fake_logmass_prob,
                sd_func=fake_projector_sd,
            )

        @pytest.fixture
        def model_physical():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_scattered_rho_total,
                comoving=False,
                logmass_prob_dist_func=fake_logmass_prob,
                sd_func=fake_projector_sd,
            )

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)


    def describe_stacked_kappa():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_scattered_rho_total,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                logmass_prob_dist_func=fake_logmass_prob,
                sd_func=fake_projector_sd,
            )

        def it_calculates_stacked_kappa_profiles(model):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model.stacked_kappa(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)

        @pytest.fixture
        def model_loop():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_scattered_rho_total,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                logmass_prob_dist_func=fake_logmass_prob,
                sd_func=fake_projector_sd,
                vectorized=False,
                num_mu_bins=4,
            )

        def the_non_vectorized_option_works(model_loop):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model_loop.stacked_kappa(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)


def describe_MatchingConvergenceModel():

    def describe_angle_scale_distance():

        @pytest.fixture
        def model_comoving():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_rho_total,
                comoving=True,
                sd_func=fake_projector_sd,
            )

        @pytest.fixture
        def model_physical():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_rho_total,
                comoving=False,
                sd_func=fake_projector_sd,
            )

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)


    def describe_stacked_kappa():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                rho_func=fake_rho_total,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                sd_func=fake_projector_sd,
            )

        def it_calculates_stacked_kappa_profiles(model):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model.stacked_kappa(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)


def describe_MatchingShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            return maszcal.lensing.MatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=fake_rho_total,
                units=u.Msun/u.pc**2,
                esd_func=fake_projector_esd,
            )

        def it_calculates_stacked_delta_sigma_profiles(model):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_delta_sigma(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)


def describe_ScatteredMatchingShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = 0.5*np.random.rand(NUM_CLUSTERS)
            return maszcal.lensing.ScatteredMatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=fake_scattered_rho_total,
                esd_func=fake_projector_esd,
                logmass_prob_dist_func=fake_logmass_prob,
                units=u.Msun/u.pc**2,
            )

        @pytest.fixture
        def model_loop():
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = 0.5*np.random.rand(NUM_CLUSTERS)
            return maszcal.lensing.ScatteredMatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=fake_scattered_rho_total,
                esd_func=fake_projector_esd,
                logmass_prob_dist_func=fake_logmass_prob,
                units=u.Msun/u.pc**2,
                vectorized=False,
                num_mu_bins=4,
            )

        def it_calculates_stacked_delta_sigma_profiles(model):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_delta_sigma(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)

        def the_non_vectorized_option_works(model_loop):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model_loop.stacked_delta_sigma(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)
