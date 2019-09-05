import numpy as np
import astropy.units as u
from maszcal.defaults import DefaultCosmology
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_astropy_cosmology


class NfwModel:
    """
    SHAPE mass, z, r, cons
    """
    def __init__(self, cosmo_params=DefaultCosmology()):
        self._delta = 200

        if isinstance(cosmo_params, DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def omega(self, z):
        return self.astropy_cosmology.efunc(z)**2

    def _reference_density(self, zs):
        """
        SHAPE z
        """
        redshift_dep = self.astropy_cosmology.Om(zs)/self.omega(zs)
        return (self.astropy_cosmology.critical_density0 * redshift_dep).to(u.Msun/u.Mpc**3).value

    def _radius_delta(self, zs, masses):
        """
        SHAPE mass, z
        """
        pref = 3 / (4*np.pi)
        return (pref * masses[:, None] / (self._delta*self._reference_density(zs))[None, :])**(1/3)

    def _scale_radius(self, zs, masses, cons):
        """
        SHAPE mass, z, cons
        """
        return self._radius_delta(zs, masses)[:, :, None]/cons[None, None, :]

    def _delta_c(self, cons):
        return (self._delta * cons**3)/(3 * (np.log(1+cons) - cons/(1+cons)))

    def _less_than_func(self, x):
        return (
            8 * np.arctanh(np.sqrt((1-x) / (1+x))) / (x**2 * np.sqrt(1 - x**2))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctanh(np.sqrt((1-x) / (1+x))) / ((x**2 - 1) * np.sqrt(1 - x**2))
        )

    def _equal_func(self, x):
        return 10/3 + 4*np.log(1/2)

    def _greater_than_func(self, x):
        return (
            8 * np.arctan(np.sqrt((x-1) / (x+1))) / (x**2 * np.sqrt(x**2 - 1))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctan(np.sqrt((x-1) / (x+1))) / ((x**2 - 1)**(3/2))
        )

    def _inequality_func(self, xs):
        less_than_mask = xs < 1
        equal_mask = xs == 1
        greater_than_mask = xs > 1

        full_func_vals = np.zeros(xs.shape)

        full_func_vals[less_than_mask] = self._less_than_func(xs[less_than_mask])
        full_func_vals[equal_mask] = self._equal_func(xs[equal_mask])
        full_func_vals[greater_than_mask] = self._greater_than_func(xs[greater_than_mask])

        return full_func_vals

    def delta_sigma(self, rs, zs, masses, cons):
        """
        SHAPE mass, z, r, cons
        """
        scale_radii = self._scale_radius(zs, masses, cons)
        prefactor = scale_radii * self._delta_c(cons)[None, None, :] * self._reference_density(zs)[None, :, None]

        xs = rs[None, None, :, None]/scale_radii[:, :, None, :]

        postfactor = self._inequality_func(xs)

        return prefactor[:, :, None, :] * postfactor
