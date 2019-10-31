import numpy as np
import scipy.optimize
from maszcal.fitting.likelihood import GaussianLikelihood


class MaxLikelihoodFitter:
    @classmethod
    def _get_log_like_func(cls, model_func, data, covariance):
        """
        Note that this function returns the _negative_ log likelihood
        """
        return lambda param: -GaussianLikelihood.log_like(param, model_func, data, covariance)

    @classmethod
    def _minimize_func(cls, func, guess):
        result = scipy.optimize.minimize(func, guess)
        return result.x

    @classmethod
    def fit(cls, model_func, guess, data, covariance):
        ln_like_func = cls._get_log_like_func(model_func, data, covariance)

        return cls._minimize_func(ln_like_func, guess)
