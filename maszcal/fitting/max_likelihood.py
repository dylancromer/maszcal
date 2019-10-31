from dataclasses import dataclass
import numpy as np
import scipy.optimize
from maszcal.fitting.likelihood import GaussianLikelihood
import maszcal.nothing as nothing


@dataclass
class MaxLikelihoodFitter:
    data: np.ndarray
    fisher: np.ndarray
    ln_prior_func: object = nothing.NoPriorFunc()

    def _neg_log_posterior(self, param, ln_like_func):
        if isinstance(self.ln_prior_func, nothing.NoPriorFunc):
            neg_ln_post = -ln_like_func(param)
        else:
            ln_prior = self.ln_prior_func(param)
            neg_ln_post = -ln_prior if (ln_prior == -np.inf) else -ln_prior - ln_like_func(param)

        return neg_ln_post

    def _get_log_like_func(self, model_func):
        return lambda param: GaussianLikelihood.log_like(param, model_func, self.data, self.fisher)

    def _minimize_func(self, func, guess):
        result = scipy.optimize.minimize(func, guess)
        return result.x

    def fit(self, model_func, guess):
        ln_like_func = self._get_log_like_func(model_func)

        neg_ln_posterior_func = lambda param: self._neg_log_posterior(param, ln_like_func)

        return self._minimize_func(neg_ln_posterior_func, guess)
