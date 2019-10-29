import pytest
import numpy as np
from maszcal.fitting.likelihood import GaussianLikelihood


def describe_gaussian_likelihood():

    @pytest.fixture
    def likelihood():
        return GaussianLikelihood()

    def it_has_a_gaussian_full_likelihood(likelihood):
        rs, data_dsigs = np.loadtxt('data/test/testdata.csv', delimiter=',').T
        data = rs*data_dsigs
        covariance = np.identity(rs.size)

        theory_func = lambda params: data
        params = np.array((3, 0))

        like = likelihood.likelihood(params, theory_func, data, covariance)

        assert like == 1 / (2*np.pi)**(params.size/2)

    def it_returns_the_appropriate_log_likelihood(likelihood):
        rs, data_dsigs = np.loadtxt('data/test/testdata.csv', delimiter=',').T
        data = rs*data_dsigs
        covariance = np.identity(rs.size)

        theory_func = lambda params: data
        params = np.array((3, 0))

        loglike = likelihood.log_like(params, theory_func, data, covariance)

        assert loglike == 0
