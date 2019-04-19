from maszcal.model import StackedModel
import numpy as np


stacked_model = StackedModel()


def test_delta_sigma_of_m():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1,2, 20)
    mus = stacked_model.mus
    cons = stacked_model.concentrations

    delta_sigmas = stacked_model.delta_sigma_of_mass(rs, mus, cons)

    precomp_delta_sigmas = np.ones((rs.size, mus.size))

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_delta_sigma_of_r():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1, 1, 40)

    delta_sigmas = stacked_model.delta_sigma(rs)

    precomp_delta_sigmas = np.ones(rs.shape)

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_power_spectrum():
    stacked_model.calc_power_spect()
