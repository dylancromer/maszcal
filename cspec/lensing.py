from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import cmaszcal.lensing
import maszcal.cosmology


def fake_projector_esd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


class FakeMatchingConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size))


def describe_MatchingBaryonShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def model(mocker):
            mocker.patch('cmaszcal.lensing.ConModel', new=FakeConModel)
            mocker.patch('cmaszcal.lensing.projector.esd', new=fake_projector_esd)
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return cmaszcal.lensing.MatchingBaryonShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
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
        def model(mocker):
            mocker.patch('cmaszcal.lensing.MatchingConModel', new=FakeMatchingConModel)
            mocker.patch('cmaszcal.lensing.projector.esd', new=fake_projector_esd)
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return cmaszcal.lensing.MatchingCmBaryonShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
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
