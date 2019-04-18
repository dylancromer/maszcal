from maszcal.model import StackedModel
import numpy as np

def test_delta_sigma_of_r():
    stacked_model = StackedModel()

    rs = np.logspace(-1, 1, 40)

    delta_sigmas = stacked_model.delta_sigma(rs)

    precomp_delta_sigmas = np.ones(rs.shape)

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)
