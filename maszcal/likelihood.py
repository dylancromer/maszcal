from dataclasses import dataclass
import numpy as np


class GaussianLikelihood:
    def __init__(self, radii, data, theory_func, covariance):
        self.radii = radii
        self.data = data
        self.theory_func = theory_func
        self.covariance = covariance

    def calc_likelihood(self, params):
        n = params.size
        prefac = 1 / (2*np.pi)**(n/2)
        prefac *= np.sqrt(np.linalg.det(self.covariance))

        theory = self.theory_func(params)
        diff = self.data - theory
        fisher = np.linalg.inv(self.covariance)

        return prefac * np.exp(-0.5 * diff.T @ fisher @ diff)

    def calc_loglike(self, params):
        theory = self.theory_func(params)
        diff = self.data - theory
        fisher = np.linalg.inv(self.covariance)

        return -0.5 * diff.T @ fisher @ diff
