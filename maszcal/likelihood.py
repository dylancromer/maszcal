import numpy as np


class GaussianLikelihood:
    def get_theory(self):
        pass

    def calc_likelihood(self, params):
        n = params.size
        prefac = 1 / (2*np.pi)**(n/2)
        prefac *= np.sqrt(np.linalg.det(self.covariance))

        theory = self.get_theory(params)
        diff = self.data - theory
        fisher = np.linalg.inv(self.covariance)

        return prefac * np.exp(-0.5 * diff.T @ fisher @ diff)

    def calc_loglike(self, params):
        theory = self.get_theory(params)
        diff = self.data - theory
        fisher = np.linalg.inv(self.covariance)

        return -0.5 * diff.T @ fisher @ diff
