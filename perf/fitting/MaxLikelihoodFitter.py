
import pytest
import numpy as np
from maszcal.fitting import MaxLikelihoodFitter


def describe_max_likelihood_fitter():

    def describe_fit():

        @pytest.fixture
        def fitter():
            data = 2*np.linspace(0, 1, 10)
            fisher = np.identity(10)
            return MaxLikelihoodFitter(data=data, fisher=fisher)

        @pytest.fixture
        def model_func():
            return lambda a: a*np.linspace(0, 1, 10)

        def its_fast(fitter, model_func, benchmark):
            benchmark(fitter.fit, model_func, 0.5)
