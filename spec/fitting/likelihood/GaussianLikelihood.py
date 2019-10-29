import pytest
import numpy as np
from maszcal.fitting.likelihood import GaussianLikelihood


def describe_gaussian_likelihood():

    @pytest.fixture
    def likelihood():
        rs, data_dsigs = np.loadtxt('data/test/testdata.csv', delimiter=',').T
        data = rs*data_dsigs
        covariance = np.identity(rs.size)

        theory_func = lambda params: data

        return GaussianLikelihood(radii=rs,
                                  data=data,
                                  theory_func=theory_func,
                                  covariance=covariance)

    def it_has_a_gaussian_full_likelihood(likelihood):
        params = np.array((3, 0))

        like = likelihood.calc_likelihood(params)

        assert like == 1 / (2*np.pi)**(params.size/2)

    def it_returns_the_appropriate_log_likelihood(likelihood):
        params = np.array((3, 0))

        loglike = likelihood.calc_loglike(params)

        assert loglike == 0
