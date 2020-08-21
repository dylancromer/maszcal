import numpy as np
import maszcal.mathutils


class MiscenteringDistributions:
    @staticmethod
    def rayleigh_dist(x, scale):
        scale = maszcal.mathutils.atleast_kd(scale, x.ndim+1, append_dims=False)
        x = x[..., None]
        return (x/scale**2) * np.exp(-(x/scale)**2 / 2)


class SzMassDistributions:
    @staticmethod
    def lognormal_dist(mu, mu_sz, a_sz, b_sz, scatter):
        mu, mu_sz, a_sz, b_sz, scatter = maszcal.mathutils.expand_parameter_dims(mu, mu_sz, a_sz, b_sz, scatter)
        prefac = 1 / (np.sqrt(2*np.pi) * scatter)
        diff = mu_sz - b_sz*mu - a_sz
        return prefac * np.exp(-diff**2/(2*scatter**2))
