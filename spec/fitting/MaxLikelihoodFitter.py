import pytest
import numpy as np
from maszcal.fitting import MaxLikelihoodFitter


def describe_max_likelihood_fitter():

    def describe_fit():

        @pytest.fixture
        def fitter():
            data = 2*np.linspace(0, 1, 10)
            fish = np.identity(10)
            return MaxLikelihoodFitter(data=data, fisher=fish)

        def it_returns_the_correct_fit(fitter):
            guess = 0.5
            def model_func(a): return a*np.linspace(0, 1, 10)

            best_fit_param = fitter.fit(model_func, guess)

            assert np.allclose(best_fit_param, 2)

        def it_can_use_a_prior_on_the_params():
            data = 2*np.linspace(0, 1, 10)
            fish = np.identity(10)
            def prior(a): return -np.inf if a < 0 else 0

            fitter = MaxLikelihoodFitter(data=data, fisher=fish, ln_prior_func=prior)

            guess = 0.5
            def model_func(a): return a*np.linspace(0, 1, 10)

            best_fit_param = fitter.fit(model_func, guess)

            assert np.allclose(best_fit_param, 2)
