import numpy as np


class GaussianLikelihood:
    @classmethod
    def likelihood(cls, params, theory_func, data, covariance, fisher):
        n = params.size
        prefac = 1 / (2*np.pi)**(n/2)
        prefac *= np.sqrt(np.linalg.det(covariance))

        theory = theory_func(params)
        diff = data - theory

        return prefac * np.exp(-0.5 * diff.T @ fisher @ diff)

    @classmethod
    def log_like(cls, params, theory_func, data, fisher):
        theory = theory_func(params)
        diff = data - theory
        fisher = np.linalg.inv(fisher)

        return -0.5 * diff.T @ fisher @ diff
