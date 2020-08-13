import numpy as np
import maszcal.mathutils


class MiscenteringDistributions:
    @staticmethod
    def rayleigh_dist(x, scale):
        scale = maszcal.mathutils.atleast_kd(scale, x.ndim+1, append_dims=False)
        x = x[..., None]
        return (x/scale**2) * np.exp(-(x/scale)**2 / 2)
