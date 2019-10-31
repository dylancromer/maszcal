import numpy as np
import scipy.optimize
from maszcal.fitting.likelihood import GaussianLikelihood
import maszcal.nothing as nothing


class MaxLikelihoodFitter:
    @classmethod
    def _get_log_like_func(cls, model_func, data, covariance):
        return lambda param: GaussianLikelihood.log_like(param, model_func, data, covariance)

    @classmethod
    def _minimize_func(cls, func, guess):
        result = scipy.optimize.minimize(func, guess)
        return result.x

    @classmethod
    def fit(cls, model_func, guess, data, covariance, ln_prior_func=nothing.NoPriorFunc()):
        ln_like_func = cls._get_log_like_func(model_func, data, covariance)

        if isinstance(ln_prior_func, nothing.NoPriorFunc):
            ln_posterior_func = lambda param: -ln_like_func(param)
        else:
            ln_posterior_func = lambda param: -ln_like_func(param) - ln_prior_func(param)

        return cls._minimize_func(ln_posterior_func, guess)
