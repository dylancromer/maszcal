import pytest
import numpy as np
import maszcal.likelihoods


def describe_log_gaussian_shape():

    def it_calculates_a_log_gaussian_likehood_shape():
        model = np.random.rand(10)
        cov = np.identity(10)/2
        data = np.random.rand(10)

        diff = model - data
        fisher = np.linalg.inv(cov)
        should_be = -(diff @ fisher @ diff.T)/2

        actually_is = maszcal.likelihoods.log_gaussian_shape(model, data, cov)

        assert np.all(should_be == actually_is)
