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
            fish = np.identity(10)

            best_fit_param = fitter.fit(model_func, guess, data, fish)

            assert np.allclose(best_fit_param, 1)

        def it_returns_the_correct_fit(fitter):
            guess = 0.5
            def model_func(a): return a*np.linspace(0, 1, 10)
            data = 2*np.linspace(0, 1, 10)
            fish = np.identity(10)

            best_fit_param = fitter.fit(model_func, guess, data, fish)

            assert np.allclose(best_fit_param, 2)

        def it_can_use_a_prior_on_the_params(fitter):
            guess = 0.5
            def model_func(a): return a*np.linspace(0, 1, 10)
            def prior(a): return -np.inf if a < 0 else 0
            data = 2*np.linspace(0, 1, 10)
            fish = np.identity(10)

            best_fit_param = fitter.fit(model_func, guess, data, fish, ln_prior_func=prior)

            assert np.allclose(best_fit_param, 2)
