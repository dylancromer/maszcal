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


class FakeMatchingConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size))


def describe_MatchingBaryonConvergenceModel():

    def describe_stacked_kappa():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingBaryonConvergenceModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                sd_func=fake_projector_sd,
            )

        def it_calculates_stacked_kappa_profiles(model):
            thetas = np.logspace(-4, np.log(15 * (2*np.pi/360)/(60)), 8)
            cons = 2*np.ones(2)
            alphas = np.ones(2)
            betas = np.ones(2)
            gammas = np.ones(2)
            a_szs = np.array([-1, 0, 1])

            sds = model.stacked_kappa(thetas, cons, alphas, betas, gammas, a_szs)

            assert np.all(sds >= 0)
            assert sds.shape == (3, 8, 2)


def describe_MatchingBaryonShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingBaryonShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                esd_func=fake_projector_esd,
            )

        def it_calculates_stacked_delta_sigma_profiles(model):
            rs = np.logspace(-1, 1, 8)
            cons = 2*np.ones(2)
            alphas = np.ones(2)
            betas = np.ones(2)
            gammas = np.ones(2)
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)

            assert np.all(esds >= 0)
            assert esds.shape == (3, 8, 2)


def describe_MatchingCmBaryonShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model():
            NUM_CLUSTERS = 10
            return maszcal.lensing.MatchingCmBaryonShearModel(
                sz_masses=2e13*np.random.randn(NUM_CLUSTERS) + 2e14,
                redshifts=np.random.rand(NUM_CLUSTERS),
                lensing_weights=np.random.rand(NUM_CLUSTERS),
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                con_class=FakeMatchingConModel,
                esd_func=fake_projector_esd,
            )

        def it_calculates_stacked_delta_sigma_profiles(model):
            rs = np.logspace(-1, 1, 8)
            alphas = np.ones(2)
            betas = np.ones(2)
            gammas = np.ones(2)
            a_szs = np.array([-1, 0, 1])

            esds = model.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)

            assert np.all(esds >= 0)
            assert esds.shape == (3, 8, 2)
