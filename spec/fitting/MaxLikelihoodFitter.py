import pytest
import numpy as np
from maszcal.fitting import MaxLikelihoodFitter


def describe_max_likelihood_fitter():

    def describe_fit():

        @pytest.fixture
        def fitter():
            data = np.ones(10)
            def model_func(a): return a*np.ones(10)
            cov = np.identity(10)

            return MaxLikelihoodFitter(data=data, model_func=model_func, covariance=cov)

        def it_returns_the_correct_fit(fitter):
            best_fit_param = fitter.fit()

            assert np.allclose(best_fit_param, 1)
