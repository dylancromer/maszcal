import numpy as np
from maszcal.likelihood import GaussianLikelihood


def describe_gaussian_likelihood():

    def it_has_a_gaussian_full_likelihood():
        likelihood = GaussianLikelihood()

        # r, c, a_sz
        params = np.array((1, 3, 0))
        rs, data_dsigs = np.loadtxt('data/test/testdata.csv', delimiter=',').T
        data = rs*data_dsigs

        likelihood.radii = rs
        likelihood.data = data
        likelihood.covariance = np.identity(rs.size)

        likelihood.get_theory = lambda p: data

        like = likelihood.calc_likelihood(params)

        assert like == 1 / (2*np.pi)**(params.size/2)

    def it_returns_the_appropriate_log_likelihood():
        likelihood = GaussianLikelihood()

        # r, c, a_sz
        params = np.array((1, 3, 0))
        rs, data_dsigs = np.loadtxt('data/test/testdata.csv', delimiter=',').T
        data = rs*data_dsigs

        likelihood.radii = rs
        likelihood.data = data
        likelihood.covariance = np.identity(rs.size)

        likelihood.get_theory = lambda p: data

        loglike = likelihood.calc_loglike(params)

        assert loglike == 0
