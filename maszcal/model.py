import numpy as np

class StackedModel():
    def __init__(self):
        self.sigma_muszmu = 0.2
        pass

    def mu_sz(self, mus, as_, bs):
        return bs*mus + as_

    def prob_musz_given_mu(self, mus, as_, bs):
        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        mu_szs = self.mu_sz(mus, as_, bs)

        exps = np.exp(-(mu_szs - mus - as_)**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps
