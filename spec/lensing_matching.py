from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology


def fake_scattered_lensing_func(rs, zs, mus, *params):
    return np.ones(rs.shape[0:1] + mus.shape + zs.shape + (params[0].size,))


def fake_lensing_func(rs, zs, mus, *params):
    return np.ones(rs.shape[0:1] + mus.shape + (params[0].size,))


def fake_scattered_rho_total(rs, zs, mus, *params):
    return np.ones(rs.shape[0:1] + mus.shape + zs.shape + (params[0].size,))


def fake_logmass_prob(z, mu):
    return np.ones(mu.shape + z.shape)


def describe_BlockStacker():

    def describe_get_array_block():

        def it_returns_the_ith_block_of_the_input_array_for_a_given_size():
            array = np.arange(10)
            block_size = 3
            index = 1
            assert np.all(
                maszcal.lensing.BlockStacker._get_array_block(index, block_size, array) == np.array([3, 4, 5])
            )

    def describe_stacked_signal():

        @pytest.fixture
        def block_stacker():
            NUM_CLUSTERS = 103
            return maszcal.lensing.BlockStacker(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS)+0.1,
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                block_size=10,
                model=maszcal.lensing.ScatteredMatchingConvergenceModel,
                model_kwargs={
                    'lensing_func': fake_scattered_lensing_func,
                    'logmass_prob_dist_func': fake_logmass_prob,
                },
            )

        def it_creates_a_stacked_model_in_blocks_of_clusters(block_stacker):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = block_stacker.stacked_signal(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)


def describe_ScatteredMatchingConvergenceModel():

    def describe_angle_scale_distance():

        @pytest.fixture
        def model_comoving():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_scattered_lensing_func,
                comoving=True,
                logmass_prob_dist_func=fake_logmass_prob,
            )

        @pytest.fixture
        def model_physical():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_scattered_lensing_func,
                comoving=False,
                logmass_prob_dist_func=fake_logmass_prob,
            )

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)


    def describe_stacked_convergence():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_scattered_lensing_func,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                logmass_prob_dist_func=fake_logmass_prob,
            )

        def it_calculates_stacked_convergence_profiles(model):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model.stacked_convergence(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)

        @pytest.fixture
        def model_loop():
            NUM_CLUSTERS = 10
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_scattered_lensing_func,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                logmass_prob_dist_func=fake_logmass_prob,
                vectorized=False,
                num_mu_bins=4,
            )

        def the_non_vectorized_option_works(model_loop):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model_loop.stacked_convergence(thetas, a_szs, *rho_params)

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
                lensing_func=fake_lensing_func,
                comoving=True,
            )

        @pytest.fixture
        def model_physical():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_lensing_func,
                comoving=False,
            )

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)


    def describe_stacked_convergence():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                lensing_func=fake_lensing_func,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
            )

        def it_calculates_stacked_convergence_profiles(model):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            sds = model.stacked_convergence(thetas, a_szs, *rho_params)

            assert np.all(sds >= 0)
            assert sds.shape == (8, 3, 2)


def describe_MatchingShearModel():

    def describe_stacked_excess_surface_density():

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
                lensing_func=fake_lensing_func,
                units=u.Msun/u.pc**2,
            )

        def it_calculates_stacked_excess_surface_density_profiles(model):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_excess_surface_density(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)


def describe_ScatteredMatchingShearModel():

    def describe_stacked_excess_surface_density():

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
                lensing_func=fake_scattered_lensing_func,
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
                lensing_func=fake_scattered_lensing_func,
                logmass_prob_dist_func=fake_logmass_prob,
                units=u.Msun/u.pc**2,
                vectorized=False,
                num_mu_bins=4,
            )

        def it_calculates_stacked_excess_surface_density_profiles(model):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_excess_surface_density(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)

        def the_non_vectorized_option_works(model_loop):
            rs = np.logspace(-1, 1, 8)
            rho_params = np.ones((np.random.randint(2, 10), 2))
            a_szs = np.array([-1, 0, 1])

            esds = model_loop.stacked_excess_surface_density(rs, a_szs, *rho_params)

            assert np.all(esds >= 0)
            assert esds.shape == (8, 3, 2)
