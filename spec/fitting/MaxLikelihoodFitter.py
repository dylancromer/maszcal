import pytest
import numpy as np
from maszcal.fitting import MaxLikelihoodFitter


def describe_max_likelihood_fitter():

    def describe_fit():

        @pytest.fixture
        def fitter():
            return MaxLikelihoodFitter()

        def it_returns_the_correct_fit_for_simple_data_and_model(fitter):
            def model_func(a): return a*np.ones(10)
            guess = 0.5
            data = np.ones(10)
            cov = np.identity(10)

            best_fit_param = fitter.fit(model_func, guess, data, cov)

            assert np.allclose(best_fit_param, 1)

        def it_returns_the_correct_fit(fitter):
            guess = 0.5
            def model_func(a): return a*np.linspace(0, 1, 10)
            data = 2*np.linspace(0, 1, 10)
            cov = np.identity(10)

            best_fit_param = fitter.fit(model_func, guess, data, cov)

            assert np.allclose(best_fit_param, 2)
